import time
import numpy as np
from collections import deque

SYSTEM_PROMPT = """
당신은 인터뷰 스트레스 코칭 AI입니다.
멀티모달 스트레스 감지 모델(영상+음성)의 분석 결과를 받아 면접자에게 실질적인 피드백을 제공합니다.

규칙:
1. 항상 한국어로 2~3문장 이내로 간결하게 답변하세요.
2. [주요 감지 채널] 비중이 높은 쪽을 중심으로 피드백하세요.
   - 영상 비중이 높으면 → 눈 맞춤, 고개 기울기, 입 움직임, 몸 자세 언급
   - 음성 비중이 높으면 → 말 속도, 목소리 떨림, 음량 변화 언급
3. 스트레스 수준에 따라 어조를 조절하세요.
   - 낮음(deviation<1): 격려 위주
   - 중간(1~2): 부드러운 조언
   - 높음(>2): 구체적인 안정화 방법 제시
4. 이전 피드백과 겹치지 않게 다양한 표현을 사용하세요.
5. 관찰 사실만 언급하고, 스트레스의 '원인'이라고 단정하지 마세요.
6. 관찰된 구체적 수치(눈 깜빡임 횟수, 고개 각도 등)는 자연스럽게 녹여서 사용하세요.
"""


class FeedbackEngine:
    """
    스트레스 점수 + gate_w + 관찰 특징 → LLM 자연어 피드백.

    - 개인 베이스라인 캘리브레이션 (Z-score 편차)
    - gate_w 기반 주도 채널 우선 피드백
    - MediaPipe 기반 영상 특징 활용
    - 30초 피드백 쿨다운
    - DSPy + RAG 기반 근거 있는 피드백 (api_key 있을 때)
    """

    def __init__(self, llm_client=None, task_name='Speaking',
                 window_size=10, trend_size=50):
        self.llm       = llm_client
        self.task_name = task_name
        self._init_dspy_rag(llm_client)

        self.window_size   = window_size
        self.score_history = deque(maxlen=window_size)
        self.trend_history = deque(maxlen=trend_size)

        self.baseline = None
        self.std_dev  = 5.0

        self.last_intervention_time  = 0
        self.last_intervention_score = 0
        self.is_monitoring_effect    = False
        self.intervention_results    = []

        self.feedback_history = deque(maxlen=5)

    # ── DSPy + RAG 초기화 ─────────────────────────────────────────────────────

    def _init_dspy_rag(self, llm_client):
        self.dspy_coach = None
        self.rag        = None
        if llm_client is None:
            return
        try:
            import os
            from dspy_pipeline import StressCoachModule, init_dspy
            from rag_db import PaperRAG

            api_key = (getattr(llm_client, 'api_key', None)
                       or os.environ.get('LUXIA_API_KEY'))
            if init_dspy(api_key):
                self.dspy_coach = StressCoachModule()
                self.rag        = PaperRAG()
                print("✅ DSPy + RAG 코칭 파이프라인 초기화 완료")
        except Exception as e:
            print(f"⚠️  DSPy/RAG 초기화 실패 (rule-based fallback 사용): {e}")

    def set_baseline(self):
        if len(self.trend_history) > 0:
            self.baseline = np.mean(self.trend_history)
            self.std_dev  = max(2.0, np.std(self.trend_history))
            print(f"🎯 Baseline Set: {self.baseline:.2f}")

    def set_task(self, task_name: str):
        self.task_name = task_name

    # ── 메인 처리 ──────────────────────────────────────────────────────────────

    def process_score(self, raw_score: float, gate_w: list, dominant: str,
                      audio_feat: dict = None, video_feat: dict = None,
                      concept_contrib: dict = None):
        score_pct = raw_score * 100
        self.score_history.append(score_pct)
        self.trend_history.append(score_pct)
        current_avg = sum(self.score_history) / len(self.score_history)

        if self.baseline is None:
            if len(self.trend_history) >= 15:
                self.set_baseline()
            return {
                "status": "LEARNING", "smoothed_score": round(current_avg, 1),
                "message": "베이스라인 측정 중...",
                "deviation": 0, "gate_w": gate_w, "dominant": dominant,
                "success_count": 0, "total_interventions": 0,
            }

        deviation = (current_avg - self.baseline) / self.std_dev

        effect_msg = ""
        if self.is_monitoring_effect and (time.time() - self.last_intervention_time) > 10:
            delta   = current_avg - self.last_intervention_score
            success = delta < -2.0
            self.intervention_results.append({"success": success, "delta": round(delta, 2)})
            self.is_monitoring_effect = False
            effect_msg = " (효과 있음 ✓)" if success else " (변화 없음)"

        can_intervene = (time.time() - self.last_intervention_time) > 30

        if deviation > 2.0:
            status = "HIGH"
            if can_intervene:
                message = self._generate_llm_feedback(
                    current_avg, deviation, gate_w, dominant,
                    audio_feat, video_feat, concept_contrib)
                self._trigger_intervention(current_avg)
            else:
                message = "스트레스 높음 — 잠시 후 조언 제공 예정"
        elif deviation > 1.0:
            status = "MID"
            if can_intervene:
                message = self._generate_llm_feedback(
                    current_avg, deviation, gate_w, dominant,
                    audio_feat, video_feat, concept_contrib)
                self._trigger_intervention(current_avg)
            else:
                message = "약간 긴장 상태"
        elif deviation < -1.0:
            status = "STABLE"
            message = "안정적인 상태입니다. 잘 하고 있어요!"
        else:
            status = "NORMAL"
            message = "안정 상태 유지 중"

        message += effect_msg

        return {
            "status":              status,
            "smoothed_score":      round(current_avg, 1),
            "message":             message,
            "deviation":           round(deviation, 2),
            "gate_w":              gate_w,
            "dominant":            dominant,
            "success_count":       sum(1 for r in self.intervention_results if r['success']),
            "total_interventions": len(self.intervention_results),
        }

    # ── LLM 피드백 ────────────────────────────────────────────────────────────

    def _build_obs_lines(self, gate_w, audio_feat, video_feat):
        """
        gate_w 기반으로 주도 채널을 앞에 배치한 관찰 라인 반환.
        Returns (obs_lines: list[str], modal_focus: str)
        """
        video_lines = []
        audio_lines = []

        # 영상 관찰 특징
        vf = video_feat or {}
        if vf.get('face_detected', False):
            ec_lvl   = vf.get('eye_contact_level', '?')
            ec_score = vf.get('eye_contact_score')
            ec_str   = f" (이탈도 {1 - ec_score:.0%})" if ec_score is not None else ""
            video_lines.append(f"  · 눈 맞춤: {ec_lvl}{ec_str}")
            video_lines.append(
                f"  · 눈 깜빡임: {vf.get('blink_level','?')} "
                f"({vf.get('blink_count', 0)}회 / 1초)")
            video_lines.append(
                f"  · 고개 기울기: {vf.get('head_tilt_level','?')} "
                f"({vf.get('head_tilt_deg', 0):.1f}°)")
            video_lines.append(
                f"  · 입 움직임: {vf.get('mouth_level','?')}")
        video_lines.append(
            f"  · 몸 움직임: {vf.get('motion_level','?')}")

        # 음성 관찰 특징
        af = audio_feat or {}
        if af:
            audio_lines.append(
                f"  · 말 속도: {af.get('speech_pace','?')} "
                f"(침묵 비율 {af.get('silence_ratio',0):.0%})")
            audio_lines.append(
                f"  · 목소리 안정도: {af.get('instability_level','?')}")
            audio_lines.append(
                f"  · 음량 추세: {af.get('energy_trend','?')}")

        # gate_w로 순서 결정
        video_w, audio_w = gate_w[0], gate_w[1]
        if video_w > audio_w:
            modal_focus = (f"영상 채널 {video_w*100:.0f}%가 주도 — "
                           "표정/눈 맞춤/자세 중심으로 피드백해주세요.")
            obs = video_lines + ["  ---"] + audio_lines
        else:
            modal_focus = (f"음성 채널 {audio_w*100:.0f}%가 주도 — "
                           "목소리/말투/호흡 중심으로 피드백해주세요.")
            obs = audio_lines + ["  ---"] + video_lines

        return obs, modal_focus

    def _build_stress_analysis(self, score, deviation, gate_w, dominant,
                               audio_feat, video_feat, concept_contrib) -> str:
        """DSPy / 직접 LLM 모두에서 쓰는 분석 요약 문자열."""
        obs, modal_focus = self._build_obs_lines(gate_w, audio_feat, video_feat)
        cbm_text = ""
        if concept_contrib:
            top3 = sorted(concept_contrib.items(),
                          key=lambda x: abs(x[1]), reverse=True)[:3]
            cbm_text = "  CBM: " + ", ".join(
                f"{k}({'↑' if v > 0 else '↓'}{abs(v):.2f})" for k, v in top3)
        return (
            f"스트레스 {score:.1f}점 (베이스라인 대비 {deviation:+.1f}σ) | "
            f"주도 채널: {dominant} (영상 {gate_w[0]*100:.0f}% / 음성 {gate_w[1]*100:.0f}%)\n"
            f"{cbm_text}\n"
            f"관찰: {'; '.join(obs[:4])}\n"
            f"{modal_focus}"
        )

    def _generate_llm_feedback(self, score, deviation, gate_w, dominant,
                                audio_feat=None, video_feat=None,
                                concept_contrib=None) -> str:
        # ── DSPy + RAG 우선 ──
        if self.dspy_coach is not None:
            return self._dspy_feedback(score, deviation, gate_w, dominant,
                                       audio_feat, video_feat, concept_contrib)
        # ── 직접 Anthropic API ──
        if self.llm is not None:
            return self._direct_llm_feedback(score, deviation, gate_w, dominant,
                                             audio_feat, video_feat, concept_contrib)
        return self._fallback_feedback(gate_w, audio_feat, video_feat)

    def _dspy_feedback(self, score, deviation, gate_w, dominant,
                       audio_feat, video_feat, concept_contrib) -> str:
        analysis = self._build_stress_analysis(
            score, deviation, gate_w, dominant, audio_feat, video_feat, concept_contrib)

        # RAG: 주도 채널 기반 쿼리
        query   = f"{dominant} voice stress presentation anxiety coaching"
        papers  = self.rag.retrieve(query, k=2) if self.rag else []
        context = "\n\n".join(papers) or "일반적인 스트레스 관리 원칙"

        prev_str = "\n".join(f"- {f}" for f in self.feedback_history) or "없음"

        try:
            result   = self.dspy_coach(
                stress_analysis=analysis,
                research_context=context,
                previous_feedback=prev_str,
            )
            feedback = result.feedback.strip()
            self.feedback_history.append(feedback)

            return feedback
        except Exception as e:
            print(f"DSPy error: {e}")
            return self._fallback_feedback(gate_w, audio_feat, video_feat)

    def _direct_llm_feedback(self, score, deviation, gate_w, dominant,
                              audio_feat=None, video_feat=None,
                              concept_contrib=None) -> str:
        """직접 Luxia API 호출 (DSPy 없을 때 fallback)."""
        prev     = list(self.feedback_history) or ["없음"]
        obs, modal_focus = self._build_obs_lines(gate_w, audio_feat, video_feat)
        obs_text = '\n'.join(obs)
        cbm_text = ""
        if concept_contrib:
            top3 = sorted(concept_contrib.items(),
                          key=lambda x: abs(x[1]), reverse=True)[:3]
            lines = [f"  · {k}: {'↑스트레스' if v > 0 else '↓안정'} ({v:+.2f})"
                     for k, v in top3]
            cbm_text = "\n[CBM 주요 근거]\n" + "\n".join(lines)

        context = (
            f"[현재 분석]\n"
            f"- 스트레스 점수: {score:.1f}점 (베이스라인 대비 {deviation:+.1f}σ)\n"
            f"- 주요 감지 채널: {dominant} (영상 {gate_w[0]*100:.0f}% / 음성 {gate_w[1]*100:.0f}%)\n"
            f"- 현재 과제: {self.task_name}\n\n"
            f"[관찰된 특징]\n{obs_text}\n{cbm_text}\n"
            f"[모달리티 포커스]\n{modal_focus}\n\n"
            f"[이전 피드백]\n{chr(10).join(f'- {f}' for f in prev)}\n\n"
            f"위 분석을 바탕으로 면접자에게 2~3문장의 구체적인 피드백을 생성해주세요."
        )
        try:
            response = self.llm.messages.create(
                messages=[{"role": "user", "content": context}],
                system=SYSTEM_PROMPT,
                max_tokens=150,
            )
            feedback = response.content[0].text.strip()
            self.feedback_history.append(feedback)
            return feedback
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_feedback(gate_w, audio_feat, video_feat)

    def _fallback_feedback(self, gate_w, audio_feat, video_feat) -> str:
        """LLM 없을 때 gate_w 기반 규칙 피드백."""
        vf = video_feat or {}
        af = audio_feat or {}
        video_w = gate_w[0]

        if video_w > 0.5:  # 영상 주도
            if vf.get('eye_contact_level') == '회피 중':
                return "시선이 카메라에서 많이 벗어나 있어요. 렌즈를 바라보며 말해보세요."
            if vf.get('head_tilt_level') == '많이 기울어짐':
                return f"고개가 {abs(vf.get('head_tilt_deg',0)):.0f}° 정도 기울어져 있어요. 자세를 바로 해보세요."
            if vf.get('blink_level') == '많음':
                return "눈 깜빡임이 잦아지고 있어요. 잠깐 눈을 감고 심호흡해보세요."
            if vf.get('motion_level') == '많음':
                return "몸이 많이 움직이고 있어요. 편안한 자세로 고정해보세요."
            return "표정이 긴장돼 보여요. 잠깐 어깨 힘을 빼보세요."
        else:  # 음성 주도
            pace = af.get('speech_pace', '')
            if pace == '빠름':
                return "말이 빨라지고 있어요. 문장 사이에 잠깐 멈추며 천천히 말해보세요."
            if pace == '느림':
                return "말 속도가 느려지고 있어요. 자신감 있게 또박또박 말해보세요."
            if af.get('instability_level') == '높음':
                return "목소리가 불안정해지고 있어요. 천천히 호흡하고 말해보세요."
            if af.get('energy_trend') == '점점 작아짐':
                return "목소리가 점점 작아지고 있어요. 배에 힘을 주고 크게 말해보세요."
            return "잠깐 멈추고 깊게 숨을 들이쉰 뒤 계속 말해보세요."

    def _trigger_intervention(self, current_score):
        self.last_intervention_time  = time.time()
        self.last_intervention_score = current_score
        self.is_monitoring_effect    = True

