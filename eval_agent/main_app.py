# main_app.py

import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

# Gemini 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
llm_semaphore = asyncio.Semaphore(1)

# --- 평가 노드 함수들 ---

async def star_agent_node(answer: str) -> str:
    print("--- ⭐️ STAR 평가 시작 ---")
    prompt_template = PromptTemplate.from_template("""
    아래 답변을 STAR 기법(Situation, Task, Action, Result)으로 평가하고 점수를 10점 만점으로 부여하세요. 
    구성요소가 얼마나 명확하게 드러나는지에 집중하세요.
    - 점수: [점수]/10
    - 평가: [평가 내용]
    답변: {answer}
    """)
    chain = prompt_template | llm
    
    async with llm_semaphore:
        star_eval = await chain.ainvoke({"answer": answer})
    return star_eval.content

async def logic_agent_node(answer: str) -> str:
    print("--- 🧐 논리성 평가 시작 ---")
    prompt_template = PromptTemplate.from_template("""
    아래 답변의 논리성과 명확성을 평가하고 점수를 10점 만점으로 부여하세요.
    - 논리적인 흐름을 가졌는가?
    - 주장이 명확하게 전달되는가?
    - 점수: [점수]/10
    - 평가: [평가 내용]
    답변: {answer}
    """)
    chain = prompt_template | llm

    async with llm_semaphore:
        logic_eval = await chain.ainvoke({"answer": answer})
    return logic_eval.content

async def timing_agent_node(time_in_seconds: int) -> Dict[str, Any]:
    print("--- ⏱️ 발화 시간 평가 시작 ---")
    if 90 <= time_in_seconds <= 120:
        score = 10
        evaluation = "매우 적절한 발화 시간입니다."
    else:
        score = max(10 - abs(time_in_seconds - 105) // 5, 0)
        evaluation = "발화 시간이 적정 범위를 벗어났습니다."
    
    return {"score": score, "evaluation": evaluation}

async def final_report_node(eval_results: Dict[str, Any]) -> str:
    print("--- 📝 최종 보고서 생성 시작 ---")
    prompt_template = PromptTemplate.from_template("""
    아래는 면접 답변에 대한 여러 평가 결과입니다. 각 결과를 종합하여 하나의 최종 면접 평가 보고서를 작성해주세요.
    보고서는 점수와 평가 내용이 포함되어야 합니다.
    
    <평가 결과>
    - STAR 기법 평가: {star_eval}
    - 논리성 평가: {logic_eval}
    - 발화 시간 평가: {timing_eval}

    <최종 보고서>
    """)
    
    chain = prompt_template | llm

    async with llm_semaphore:
        final_report = await chain.ainvoke({
            "star_eval": eval_results["star_evaluation"],
            "logic_eval": eval_results["logic_evaluation"],
            "timing_eval": eval_results["timing_evaluation"]
        })
    return final_report.content

# --- 메인 실행 함수 ---
async def evaluate_full_answer(question: str, answer: str, time_in_seconds: int) -> Dict[str, Any]:
    # 평가 함수들을 순차적으로 호출
    star_eval = await star_agent_node(answer)
    logic_eval = await logic_agent_node(answer)
    timing_eval = await timing_agent_node(time_in_seconds)
    
    # 평가 결과들을 취합
    eval_results = {
        "star_evaluation": star_eval,
        "logic_evaluation": logic_eval,
        "timing_evaluation": timing_eval
    }
    
    # 최종 보고서 생성
    final_report = await final_report_node(eval_results)
    
    return {
        "question": question,
        "answer": answer,
        "time_in_seconds": time_in_seconds,
        "evaluations": eval_results,
        "final_report": final_report
    }

if __name__ == '__main__':
    # 이 파일만 단독으로 실행했을 때 테스트
    async def test_main():
        sample_answer = "팀 프로젝트에서 사용자 관리 기능을 개발했습니다. 사용자가 많아지면서 로딩이 느려져서 DB 쿼리를 최적화하는 작업을 맡았습니다. 인덱스를 추가하고 조인 방식을 변경해서 로딩 속도를 80% 개선하는 데 성공했습니다. 이 경험으로 성능 최적화의 중요성을 깨달았습니다."
        report = await evaluate_full_answer("프로젝트 경험에 대해 설명해주세요.", sample_answer, 100)
        print("\n=== 최종 보고서 ===\n", report["final_report"])

    asyncio.run(test_main())