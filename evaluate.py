import csv
import time
from pathlib import Path
import rag_pipeline
import vector_store

EVAL_INDEX=Path("vector_store/eval_index")

def check(case,reply):
    # Compare one answer with what the dataset expects
    should_refuse=case["should_refuse"]=="true"
    refused=reply["refused"]
    # retrieval: was the expected file and page among the retrieved passages?
    found=any(s["source"]==case["expected_source"] and str(s["page"])==case["expected_page"] for s in reply["sources"])
    # answer: do all expected keywords appear in the answer?
    keywords=[k for k in case["expected_keywords"].split(";") if k]
    keywords_ok=(not refused) and all(k.lower() in reply["answer"].lower() for k in keywords)
    if should_refuse:
        passed=refused
    else:
        passed=found and keywords_ok
    return {"found":found,"keywords_ok":keywords_ok,"refused":refused,"should_refuse":should_refuse,"passed":passed}

def percent(values):
    return f"{100*sum(values)/len(values):.0f}%" if values else "n/a"

def main():
    print("Indexing documents/ ...")
    vector_store.rebuild_index(Path("documents"),EVAL_INDEX)

    rows=[]
    for case in csv.DictReader(open("tests/test_cases.csv",newline="",encoding="utf-8")):
        start=time.time()
        try:
            reply=rag_pipeline.answer_question(case["question"],EVAL_INDEX)
        except ValueError as error:
            print("Stopped:",error)
            return
        result=check(case,reply)
        result["question"]=case["question"]
        result["expected"]=case["expected_source"]+" p"+case["expected_page"] if case["expected_source"] else "Not available"
        result["retrieved"]=", ".join(s["source"]+" p"+str(s["page"]) for s in reply["sources"][:2])
        result["seconds"]=round(time.time()-start,2)
        rows.append(result)
        print(("PASS" if result["passed"] else "FAIL"),"|",result["question"],"|",result["expected"],"|",result["retrieved"])

    answerable=[r for r in rows if not r["should_refuse"]]
    unanswerable=[r for r in rows if r["should_refuse"]]
    print("\nQuestions:",len(rows))
    print("Overall correct:",percent([r["passed"] for r in rows]))
    print("Retrieval accuracy (right document and page):",percent([r["found"] for r in answerable]))
    print("Answer correctness (expected keywords):",percent([r["keywords_ok"] for r in answerable]))
    print("Correct refusals (answer not in documents):",percent([r["refused"] for r in unanswerable]))
    print("Average response time:",round(sum(r["seconds"] for r in rows)/len(rows),2),"seconds")

    # save the evaluation sheet
    with open("evaluation_results.csv","w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=["question","expected","retrieved","passed","refused","seconds"],extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Saved evaluation_results.csv")

if __name__=="__main__":
    main()
