# api_server.py

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid
import uvicorn
import os
from dotenv import load_dotenv

# main_app.py의 평가 함수를 임포트합니다.
from main_app import evaluate_full_answer

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# 임시 세션 저장소 (실제 서비스에서는 Redis, DB 등을 사용)
sessions: Dict[str, Dict[str, Any]] = {}

class InterviewRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    time_in_seconds: int

class InterviewStartResponse(BaseModel):
    session_id: str
    message: str

class InterviewEndRequest(BaseModel):
    session_id: str

class InterviewEndResponse(BaseModel):
    interview_log: List[Dict[str, Any]]

@app.post("/start_interview", response_model=InterviewStartResponse)
async def start_interview():
    """면접 시뮬레이션을 시작하고 세션 ID를 발급합니다."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "interview_log": [],
    }
    return {"session_id": session_id, "message": "면접 시뮬레이션이 시작되었습니다."}

@app.post("/evaluate_answer")
async def evaluate_answer(request: InterviewRequest) -> Dict[str, Any]:
    """사용자의 답변을 평가하고 로그에 저장합니다."""
    session_state = sessions.get(request.session_id)
    if not session_state:
        return {"error": "유효하지 않은 세션 ID입니다."}

    try:
        # main_app의 함수를 호출하여 평가 로직을 실행
        current_eval_log = await evaluate_full_answer(
            request.question,
            request.answer,
            request.time_in_seconds
        )
        
        # 업데이트된 로그를 세션에 저장
        session_state["interview_log"].append(current_eval_log)
        
        # 현재 답변에 대한 최종 평가 내용만 반환
        return {"report_for_current_answer": current_eval_log["final_report"]}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": str(e)}

@app.post("/end_interview", response_model=InterviewEndResponse)
async def end_interview(request: InterviewEndRequest):
    """면접을 종료하고 모든 질문-답변 및 평가 기록을 반환합니다."""
    session_state = sessions.get(request.session_id)
    if not session_state:
        return {"error": "유효하지 않은 세션 ID입니다."}

    interview_log = session_state.get("interview_log", [])
    
    # 세션 정보 삭제
    del sessions[request.session_id]
    
    return {"interview_log": interview_log}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)