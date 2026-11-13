"""Offline re-score of stored v2 claim verdicts with the new deterministic verifier rules.
The judge's raw verdict before the quote check is not stored, so a claim stored as
UNSUPPORTED with a non-empty quote and valid_quote=False is assumed to have been
judged SUPPORTED (upper bound); everything else keeps its stored verdict."""
import json, glob, sys
sys.path.insert(0, "src")
from sec_llm_project.sec_intel.core.types import (Answer, Chunk, ClaimVerdict, FilingMetadata,
                                                  RetrievalResult, VerificationReport)
from sec_llm_project.sec_intel.generation.verifier import CitationVerifier

answers = {}
for f in sorted(glob.glob("out/provenance_16gb_v2/*/answer.json")):
    a = json.load(open(f)); v = (a.get("extra") or {}).get("verification")
    if v and v.get("claims"):
        answers[f.split("/")[-2]] = (a, v)
need = {e["chunk_id"] for _, v in answers.values() for e in v["evidence"]}
text = {}
with open("data/sec_index_qwen06/chunks.jsonl") as fh:
    for line in fh:
        d = json.loads(line); cid = d.get("chunk_id")
        if cid in need: text[cid] = d["text"]
print(f"answers with claims {len(answers)}, evidence chunks {len(need)}, found {len(text)}")

class Replay:
    name = "replay"
    def __init__(self, claims): self.claims = claims
    def judge(self, answer_text, context):
        out = []
        for c in self.claims:
            verdict = c["verdict"]
            if verdict == "UNSUPPORTED" and c.get("quote") and not c.get("valid_quote"):
                verdict = "SUPPORTED"
            out.append(ClaimVerdict(claim=c["claim"], verdict=verdict,
                                    citation=c.get("citation"), quote=c.get("quote", "")))
        return out

FLOOR = 0.6
old_pass = new_pass = 0; rows = []
tot_old = tot_new = sup_old = sup_new = meta = corr = 0
for qid, (a, v) in answers.items():
    ctx = [RetrievalResult(chunk=Chunk(chunk_id=e["chunk_id"], text=text.get(e["chunk_id"], ""),
                                       metadata=FilingMetadata(ticker="X")), score=1.0, rank=i)
           for i, e in enumerate(v["evidence"])]
    old = VerificationReport(claims=[ClaimVerdict(**{k: c[k] for k in ("claim", "verdict", "citation", "quote")})
                                     for c in v["claims"]])
    new = CitationVerifier(Replay(v["claims"])).verify(Answer(question="", text=a["text"]), ctx)
    fo = old.contradicted > 0 or old.groundedness < FLOOR
    fn = (new.total and (new.contradicted > 0 or new.groundedness < FLOOR)) or not new.total
    old_pass += not fo; new_pass += not fn
    tot_old += old.total; tot_new += new.total; sup_old += old.supported; sup_new += new.supported
    meta += len(new.claims) - new.total; corr += sum(c.citation_corrected for c in new.claims)
    rows.append((qid, round(old.groundedness, 2), round(new.groundedness, 2), not fo, not fn))
print(f"claims {tot_old} -> scored {tot_new} (meta excluded {meta}); citations re-pointed {corr}")
print(f"supported share {sup_old/tot_old:.3f} -> {sup_new/tot_new:.3f}")
print(f"answers passing the gate (no contradiction, groundedness >= {FLOOR}): {old_pass} -> {new_pass} of {len(answers)}")
flip = [r for r in rows if r[3] != r[4]]
print("flipped:", flip)
