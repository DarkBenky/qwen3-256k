import torch, torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

dev, dt = "cuda", torch.bfloat16

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
enc = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", torch_dtype=dt).to(dev)
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=dt).to(dev)
proj = nn.Linear(enc.config.hidden_size, llm.config.hidden_size).to(dev, dt)

block, nWinBlocks, nSum = 512, 4, 4


def encBlock(ids, grad=False):
    t = torch.tensor([ids], device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return enc(input_ids=t).last_hidden_state


def takeSums(h):
    if h.shape[1] <= nSum:
        return h
    pos = torch.linspace(0, h.shape[1] - 1, nSum, device=h.device).round().long()
    return h[:, pos]


def makeOpt():
    return bnb.optim.AdamW8bit([
        {"params": proj.parameters(), "lr": 1e-4},
        {"params": list(enc.parameters()) + list(llm.parameters()), "lr": 1e-5},
    ])


def trainStep(docIds, opt):
    nB = len(docIds) // block
    if nB < 3:
        return None
    k = int(torch.randint(1, nB - 1, (1,)))
    past = [docIds[i * block:(i + 1) * block] for i in range(k)]
    win = docIds[k * block:(k + 1) * block]
    with torch.no_grad():
        sums = torch.cat([takeSums(encBlock(b)) for b in past], 1)
    x = torch.cat([proj(sums), proj(encBlock(win, grad=True))], 1)
    logits = llm(inputs_embeds=x, logits_to_keep=block).logits
    labels = torch.tensor([docIds[k * block + 1:(k + 1) * block + 1]], device=dev)
    loss = nn.functional.cross_entropy(logits[0].float(), labels[0])
    loss.backward()
    nn.utils.clip_grad_norm_(list(proj.parameters()) + list(enc.parameters()) + list(llm.parameters()), 1.0)
    opt.step()
    opt.zero_grad()
    return loss.item()


class Session:
    def __init__(self):
        self.sumVecs = []
        self.winVecs = []
        self.tail = []
        self.tailIn = 0
        self.cache = DynamicCache()
        self.len = 0
        self.out = None

    @torch.no_grad()
    def rebuild(self):
        parts = [proj(x) for x in self.sumVecs + self.winVecs]
        x = torch.cat(parts, 1)
        self.cache = DynamicCache()
        self.out = llm(inputs_embeds=x, past_key_values=self.cache,
                       use_cache=True, logits_to_keep=1)
        self.len = x.shape[1]

    @torch.no_grad()
    def feedIds(self, ids):
        self.tail += ids
        pushed = False
        while len(self.tail) >= block:
            b, self.tail = self.tail[:block], self.tail[block:]
            self.winVecs.append(encBlock(b))
            if len(self.winVecs) > nWinBlocks:
                self.sumVecs.append(takeSums(self.winVecs.pop(0)))
            pushed = True
        if pushed:
            self.tailIn = 0
            self.rebuild()
        if self.tail:
            h = encBlock(self.tail)
            newStates = h[:, self.tailIn:]
            if newStates.shape[1] > 0:
                self.out = llm(inputs_embeds=proj(newStates), past_key_values=self.cache,
                               use_cache=True, logits_to_keep=1)
                self.len += newStates.shape[1]
                self.tailIn = len(self.tail)

    def feedText(self, text):
        self.feedIds(tok(text, return_tensors="pt").input_ids[0].tolist())

    @torch.no_grad()
    def genText(self, maxNew=256):
        ids = []
        for _ in range(maxNew):
            nxt = int(self.out.logits[:, -1].argmax(-1))
            ids.append(nxt)
            self.feedIds([nxt])
        return tok.decode(ids)


if __name__ == "__main__":
    s = Session()
    s.feedText(open("long_doc.txt", encoding="utf-8").read())
    print("positions:", s.len, "vram GB:", round(torch.cuda.memory_allocated() / 1e9, 2))
    print(s.genText(200))
    s.feedText("\n\nSummarize the document above.")
    print(s.genText(300))

    opt = makeOpt()
    text = open("long_doc.txt", encoding="utf-8").read()
    docIds = tok(text, add_special_tokens=False).input_ids
    for i in range(20):
        loss = trainStep(docIds, opt)
        if loss is not None:
            print(i, round(loss, 4))
