import time
import numpy as np
from collections import deque

class FeedbackEngine:
    """
    유저의 패턴을 학습하고, AI 조언의 효과를 실시간으로 측정하는 검증형 에이전트 엔진.
    """
    def __init__(self, window_size=10, trend_size=50):
        self.window_size = window_size
        self.score_history = deque(maxlen=window_size)
        self.trend_history = deque(maxlen=trend_size)
        
        self.baseline = None
        self.std_dev = 5.0
        
        # --- 개입 효과 측정용 변수 ---
        self.last_intervention_time = 0
        self.last_intervention_type = None
        self.last_intervention_score = 0
        self.intervention_results = [] # [{type, time, success, delta}, ...]
        self.is_monitoring_effect = False

    def set_baseline(self):
        if len(self.trend_history) > 0:
            self.baseline = np.mean(self.trend_history)
            self.std_dev = max(2.0, np.std(self.trend_history))
            print(f"🎯 Baseline Set: {self.baseline:.2f}")

    def process_score(self, raw_score):
        self.score_history.append(raw_score)
        self.trend_history.append(raw_score)
        current_avg = sum(self.score_history) / len(self.score_history)
        
        if self.baseline is None:
            if len(self.trend_history) >= 15: self.set_baseline()
            return {"status": "LEARNING", "message": "Observing baseline...", "smoothed_score": round(current_avg, 1)}

        deviation = (current_avg - self.baseline) / self.std_dev
        
        # 1. 개입 효과 모니터링 (조언 후 10초간 수치 변화 관찰)
        effect_msg = ""
        if self.is_monitoring_effect and (time.time() - self.last_intervention_time) > 10:
            delta = current_avg - self.last_intervention_score
            success = delta < -2.0 # 수치가 2점 이상 떨어지면 성공
            self.intervention_results.append({
                "type": self.last_intervention_type,
                "success": success,
                "delta": round(delta, 2)
            })
            self.is_monitoring_effect = False
            if success: effect_msg = " (Success! Your stress dropped)"
            else: effect_msg = " (No change detected)"

        # 2. 상태 결정 및 조언 트리거
        status = "NORMAL"
        message = "Maintaining stable state."
        
        # 30초마다 한 번만 조언하도록 제한 (너무 잦으면 스트레스 유발)
        can_intervene = (time.time() - self.last_intervention_time) > 30

        if deviation > 2.0:
            status = "HIGH"
            message = "Stress high. Try taking a deep breath."
            if can_intervene: self._trigger_intervention("BREATH", current_avg)
        elif deviation > 1.0:
            status = "MID"
            message = "Feeling tense? Try to speak slower."
            if can_intervene: self._trigger_intervention("PACE", current_avg)
        elif deviation < -1.0:
            status = "STABLE"
            message = "Perfect pace. Keep it up!"

        return {
            "raw_score": raw_score,
            "smoothed_score": round(current_avg, 1),
            "status": status,
            "message": message + effect_msg,
            "deviation": round(deviation, 2),
            "success_count": sum(1 for r in self.intervention_results if r['success']),
            "total_interventions": len(self.intervention_results)
        }

    def _trigger_intervention(self, tip_type, current_score):
        self.last_intervention_time = time.time()
        self.last_intervention_type = tip_type
        self.last_intervention_score = current_score
        self.is_monitoring_effect = True
        print(f"📢 AI Intervention Triggered: {tip_type}")
