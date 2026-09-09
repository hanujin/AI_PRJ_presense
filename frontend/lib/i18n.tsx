"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type Lang = "ko" | "en";

const STORAGE_KEY = "presense-lang";
const DEFAULT_LANG: Lang = "en";

// Flat key → { ko, en }. Use t("key") to read the current language.
const dict: Record<string, { ko: string; en: string }> = {
  // ── App shell ──
  "brand.tagline": { ko: "발표 AI", en: "Presentation AI" },
  "nav.dashboard": { ko: "대시보드", en: "Dashboard" },
  "nav.history": { ko: "기록", en: "History" },
  "nav.practice": { ko: "연습", en: "Practice" },
  "nav.settings": { ko: "설정", en: "Settings" },
  "cta.startPractice": { ko: "연습 시작", en: "Start Practice" },
  "header.search": { ko: "세션 검색...", en: "Search sessions..." },
  "menu.settings": { ko: "설정", en: "Settings" },
  "menu.signOut": { ko: "로그아웃", en: "Sign out" },
  "menu.login": { ko: "로그인", en: "Log in" },
  "common.loadingWorkspace": { ko: "작업공간을 불러오는 중...", en: "Loading your workspace..." },
  "lang.toggle": { ko: "언어 전환", en: "Toggle language" },

  // ── Auth ──
  "auth.headline": { ko: "더 똑똑하게 연습하고,\n더 잘 발표하세요.", en: "Practice smarter, present better." },
  "auth.desc": {
    ko: "웹캠과 마이크만으로 실시간 스트레스 감지하는 플랫폼",
    en: "Real-time stress detection platform\nfrom your webcam and microphone",
  },
  "auth.feature1": { ko: "실시간 카메라 + 마이크 분석", en: "Live camera + microphone analysis" },
  "auth.feature2": { ko: "세션마다 AI 스트레스 감지", en: "AI stress detection after each session" },
  "auth.feature3": { ko: "세션 기록 및 진행 상황 추적", en: "Session history and progress tracking" },
  "auth.welcomeBack": { ko: "다시 오신 걸 환영합니다", en: "Welcome back" },
  "auth.createAccount": { ko: "계정 만들기", en: "Create account" },
  "auth.signInSub": { ko: "PreSense 계정으로 로그인하세요.", en: "Sign in to your PreSense account." },
  "auth.signUpSub": { ko: "발표 연습 여정을 시작하세요.", en: "Start your presentation practice journey." },
  "auth.login": { ko: "로그인", en: "Login" },
  "auth.signUp": { ko: "회원가입", en: "Sign Up" },
  "auth.forgotIntro": { ko: "계정 이메일을 입력하면 비밀번호 복구 링크를 보내드립니다.", en: "Enter your account email and we will send a password recovery link." },
  "auth.name": { ko: "이름", en: "Name" },
  "auth.namePlaceholder": { ko: "성함을 입력하세요", en: "Your full name" },
  "auth.email": { ko: "이메일", en: "Email" },
  "auth.password": { ko: "비밀번호", en: "Password" },
  "auth.passwordPlaceholder": { ko: "비밀번호 입력", en: "Enter password" },
  "auth.forgotPassword": { ko: "비밀번호를 잊으셨나요?", en: "Forgot password?" },
  "auth.sendRecovery": { ko: "복구 이메일 보내기", en: "Send Recovery Email" },
  "auth.sending": { ko: "보내는 중…", en: "Sending…" },
  "auth.backToLogin": { ko: "← 로그인으로 돌아가기", en: "← Back to login" },
  "auth.notConfigured": { ko: "Supabase 미설정 — 인증이 비활성화되었습니다.", en: "Supabase not configured — auth is disabled." },
  "auth.pleaseWait": { ko: "잠시만 기다려 주세요...", en: "Please wait..." },
  "auth.enterDashboard": { ko: "로그인", en: "Enter" },
  "auth.createAccountBtn": { ko: "계정 생성", en: "Create Account" },
  "auth.recoverySent": { ko: "복구 이메일을 보냈습니다. 받은편지함에서 링크를 확인하세요.", en: "Recovery email sent. Check your inbox and click the link." },
  "auth.recoveryFailed": { ko: "복구 이메일 전송에 실패했습니다.", en: "Failed to send recovery email." },
  "auth.created": { ko: "계정이 생성되었습니다. 로그인 전 이메일에서 확인해 주세요.", en: "Account created. Check your email to confirm before logging in." },
  "auth.failed": { ko: "인증에 실패했습니다.", en: "Authentication failed." },

  // ── Settings ──
  "settings.title": { ko: "설정", en: "Settings" },
  "settings.subtitle": {
    ko: "연습 환경, 개인정보, AI 코칭 옵션을 관리하세요.",
    en: "Manage your practice preferences, privacy, and AI coaching options.",
  },
  "settings.savedBrowserOnly": { ko: "이 브라우저에만 저장되었습니다.", en: "Saved in this browser only." },
  "settings.savedAccount": { ko: "계정에 저장되었습니다.", en: "Saved to your account." },
  "settings.failedLoad": { ko: "설정을 불러오지 못했습니다.", en: "Failed to load settings." },
  "settings.failedSave": { ko: "설정을 저장하지 못했습니다.", en: "Failed to save settings." },
  "settings.clickToToggle": { ko: "클릭하여 전환", en: "Click to toggle" },
  "settings.language": { ko: "언어", en: "Language" },
  "settings.languageDesc": { ko: "인터페이스 표시 언어", en: "Interface display language" },
  "toggle.enabled": { ko: "켜짐", en: "Enabled" },
  "toggle.disabled": { ko: "꺼짐", en: "Disabled" },
  "toggle.on": { ko: "켜짐", en: "On" },
  "toggle.off": { ko: "꺼짐", en: "Off" },

  // Settings — primary toggles (by id)
  "set.camera-preview": { ko: "기본으로 실시간 카메라 미리보기", en: "Default to live camera preview" },
  "set.microphone-active-only": { ko: "활성 세션 중에만 캡처", en: "Only capture during active sessions" },
  "set.session-summary-notice": { ko: "세션마다 요약 표시", en: "Show summary after each session" },
  // Settings — section titles
  "set.section.Session Preferences": { ko: "세션 환경설정", en: "Session Preferences" },
  "set.section.Privacy Controls": { ko: "개인정보 설정", en: "Privacy Controls" },
  "set.section.AI Agent Preferences": { ko: "AI 에이전트 설정", en: "AI Agent Preferences" },
  // Settings — section items (by id)
  "set.default-camera-view": { ko: "연습 시작 시 기본 카메라 화면", en: "Default camera view on practice start" },
  "set.mute-live-feedback": { ko: "세션 종료 전까지 실시간 피드백 음소거", en: "Mute live feedback until session ends" },
  "set.auto-ai-summary": { ko: "세션 후 AI 코칭 요약 자동 생성", en: "Generate post-session AI coaching summary automatically" },
  "set.store-summaries-only": { ko: "세션 요약만 저장", en: "Store session summaries only" },
  "set.disable-frame-storage": { ko: "원본 카메라 프레임 저장 안 함", en: "Do not save raw camera preview frames" },
  "set.mic-active-practice-only": { ko: "활성 연습 중에만 마이크 접근 허용", en: "Allow microphone access only during active practice" },
  "set.concise-coaching-tone": { ko: "간결한 코칭 톤 사용", en: "Use concise coaching tone" },
  "set.prioritize-presentation-advice": { ko: "발표 관련 조언 우선", en: "Prioritize presentation-specific advice" },
  "set.future-api-key-connection": { ko: "향후 API 키 연결 준비", en: "Prepare for future API key connection" },

  // ── Dashboard ──
  "dash.title": { ko: "시스템 개요", en: "System Overview" },
  "dash.subtitle": { ko: "발표 연습으로부터의 실시간 인사이트.", en: "Real-time intelligence from your presentation practice." },
  "kpi.totalSessions": { ko: "전체 세션", en: "Total Sessions" },
  "kpi.totalSessions.synced": { ko: "계정에 저장됨", en: "Saved to your account" },
  "kpi.totalSessions.local": { ko: "Supabase 연결 시 동기화", en: "Add Supabase to sync" },
  "kpi.latestAvgStress": { ko: "최근 평균 스트레스", en: "Latest Avg Stress" },
  "kpi.scoreOutOf": { ko: "1.00 만점 점수", en: "Score out of 1.00" },
  "kpi.currentFocus": { ko: "현재 집중 영역", en: "Current Focus Area" },
  "kpi.currentFocusSub": { ko: "가장 주의가 필요한 맥락", en: "Context needing the most attention" },
  "kpi.latestDiagnosis": { ko: "최근 진단", en: "Latest Diagnosis" },
  "kpi.latestDiagnosisSub": { ko: "AI 코칭 요약", en: "AI coaching summary" },
  "badge.synced": { ko: "동기화됨", en: "Synced" },
  "badge.local": { ko: "로컬", en: "Local" },
  "badge.latest": { ko: "최근", en: "Latest" },
  "panel.stressTrend": { ko: "시간별 스트레스 추이", en: "Stress Trend Over Time" },
  "panel.stressTrendSub": { ko: "세션별 평균 스트레스 점수", en: "Average stress score per session" },
  "legend.stressScore": { ko: "스트레스 점수", en: "Stress score" },
  "chart.oldest": { ko: "가장 오래됨", en: "Oldest" },
  "chart.latest": { ko: "최근", en: "Latest" },
  "panel.sessionsByScene": { ko: "장면별 세션", en: "Sessions by Scene" },
  "panel.sessionsBySceneSub": { ko: "연습 모드 분포", en: "Practice mode breakdown" },
  "state.loadingSessions": { ko: "세션 불러오는 중...", en: "Loading sessions..." },
  "state.noSessionsTrend": { ko: "아직 세션이 없습니다 — 연습을 끝내면 추이가 표시됩니다.", en: "No sessions yet — finish a practice run to see your trend." },
  "state.noSessionsScene": { ko: "아직 세션이 없습니다 — 연습이 장면별로 여기 모입니다.", en: "No sessions yet — practice runs will be grouped by scene here." },
  "unit.session": { ko: "세션", en: "session" },
  "unit.sessions": { ko: "세션", en: "sessions" },
  "panel.recentSessions": { ko: "최근 세션", en: "Recent Sessions" },
  "panel.recentSessionsSub": { ko: "세션 기록 및 결과", en: "Session history & results" },
  "table.date": { ko: "날짜", en: "Date" },
  "table.title": { ko: "제목", en: "Title" },
  "table.duration": { ko: "시간", en: "Duration" },
  "table.avgStress": { ko: "평균 스트레스", en: "Avg Stress" },
  "table.result": { ko: "결과", en: "Result" },
  "table.action": { ko: "동작", en: "Action" },
  "table.review": { ko: "리뷰", en: "Review" },
  "table.loading": { ko: "불러오는 중...", en: "Loading..." },
  "table.noSessions": { ko: "아직 세션이 없습니다. 연습을 끝내면 여기 표시됩니다.", en: "No sessions yet. Finish a practice run to see it here." },

  // ── Review modal ──
  "modal.sessionReview": { ko: "세션 리뷰", en: "Session Review" },
  "modal.avgStressScore": { ko: "평균 스트레스 점수", en: "Average Stress Score" },
  "modal.outOfHigh": { ko: "1.00 만점 — 높은 압박", en: "Out of 1.00 — High pressure" },
  "modal.outOfModerate": { ko: "1.00 만점 — 보통 압박", en: "Out of 1.00 — Moderate pressure" },
  "modal.outOfLow": { ko: "1.00 만점 — 낮은 압박", en: "Out of 1.00 — Low pressure" },
  "modal.stressPhaseBreakdown": { ko: "단계별 스트레스 분석", en: "Stress Phase Breakdown" },
  "modal.aiDiagnosis": { ko: "AI 진단", en: "AI Diagnosis" },
  "modal.overallJudgment": { ko: "종합 판단", en: "Overall Judgment" },
  "modal.peakPressureAnalysis": { ko: "최고 압박 분석", en: "Peak Pressure Analysis" },
  "modal.noDiagnosis": { ko: "기록된 진단이 없습니다.", en: "No diagnosis recorded." },
  "modal.recommendedNextStep": { ko: "추천 다음 단계", en: "Recommended Next Step" },
  "modal.coachingInsight": { ko: "코칭 인사이트", en: "Coaching Insight" },
  "modal.coach.high": {
    ko: "높은 스트레스 감지 — 주요 포인트 사이에서 속도를 늦추는 데 집중하세요. 핵심 용어 뒤에 1~2초의 의도적인 쉼을 넣어 청중이 이해하고 본인도 호흡을 가다듬을 시간을 주세요.",
    en: "High stress detected — focus on slowing down between major points. Insert deliberate pauses (1–2 seconds) after each key term to allow the audience to absorb information and give yourself time to reset.",
  },
  "modal.coach.moderate": {
    ko: "보통 스트레스 감지 — 기본은 괜찮지만 눈에 띄는 압박 구간이 있었습니다. 긴장이 쌓이기 쉬운 중반 핵심 내용에서 일정한 속도를 유지하는 데 집중하세요.",
    en: "Moderate stress detected — your baseline is acceptable but there were noticeable pressure spikes. Work on keeping a consistent pace in the mid-session core content where tension tends to build.",
  },
  "modal.coach.low": {
    ko: "낮은 스트레스 감지 — 전달이 안정적이었습니다. 더 발전하려면 표현 변화에 집중하세요: 핵심 용어를 의도적으로 강조하면 불안을 키우지 않고도 발표가 더 생동감 있어집니다.",
    en: "Low stress detected — delivery was stable. To improve further, focus on expressive variation: deliberate emphasis on key terms will make your presentation more engaging without raising anxiety.",
  },
  "sev.high": { ko: "높음", en: "High" },
  "sev.moderate": { ko: "보통", en: "Moderate" },
  "sev.low": { ko: "낮음", en: "Low" },
  "diag.latestSessionResult": { ko: "최근 세션 결과", en: "Latest session result" },
  "diag.averageStress": { ko: "평균 스트레스", en: "Average stress" },
  "diag.nextCoachingFocus": { ko: "다음 코칭 집중 영역", en: "Next coaching focus" },
  "diag.generatedFromSession": { ko: "AI 진단", en: "AI diagnosis" },
  "diag.noDeck": { ko: "사용한 자료 없음", en: "No deck used" },
  "diag.phase.open": { ko: "도입", en: "Open" },
  "diag.phase.build": { ko: "전개", en: "Build" },
  "diag.phase.core": { ko: "핵심", en: "Core" },
  "diag.phase.close": { ko: "마무리", en: "Close" },
  "diag.peakPressure": { ko: "최고 압박은 {phase} 단계에서 {value}%까지 높아졌습니다. 자료: {deck}.", en: "Peak pressure reached {value}% around the {phase} phase. Deck: {deck}." },
  "diag.noticeablePressure": { ko: "{phase} 구간에서 눈에 띄는 압박", en: "Noticeable {phase}-section pressure" },
  "diag.stableDelivery": { ko: "전반적으로 안정적인 발표", en: "Mostly stable delivery" },
  "diag.nextAction": { ko: "주요 설명 구간 사이의 전환을 더 천천히 연습하고, 핵심 용어마다 짧게 멈춰 보세요.", en: "Practice slower transitions between your main explanation blocks and leave a short pause after each key term." },

  // ── Practice ──
  "practice.title": { ko: "연습 세션", en: "Practice Session" },
  "practice.subtitle": {
    ko: "슬라이드, 청중 압박, 조용한 세션 후 분석으로 연습하세요.",
    en: "Rehearse with slides, audience pressure, and quiet post-session analysis.",
  },
  "btn.endSession": { ko: "세션 종료", en: "End Session" },
  "btn.startSession": { ko: "세션 시작", en: "Start Session" },
  "setup.rehearsalDeck": { ko: "리허설 자료", en: "Rehearsal Deck" },
  "setup.noDeck": { ko: "선택된 자료 없음", en: "No deck selected" },
  "setup.ready": { ko: "무대 미리보기 준비 완료.", en: "Ready for stage preview." },
  "setup.needsPdf": { ko: "미리보기는 PDF 변환이 필요합니다.", en: "Preview requires PDF conversion." },
  "setup.uploadOrSample": { ko: "PDF를 업로드하거나 샘플 프레젠테이션을 사용하세요.", en: "Upload a PDF or use the sample presentation." },
  "btn.uploadDeck": { ko: "자료 업로드", en: "Upload Deck" },
  "btn.useSample": { ko: "샘플 사용", en: "Use Sample" },
  "mode.label": { ko: "피드백 표시", en: "Feedback display" },
  "mode.silent": { ko: "숨김", en: "Hidden" },
  "mode.gentle": { ko: "항상 표시", en: "Always" },
  "mode.active": { ko: "주의 시 표시", en: "On alert" },
  "signal.sessionMode": { ko: "세션 모드", en: "Session Mode" },
  "signal.elapsed": { ko: "경과 시간", en: "Elapsed Time" },
  "signal.mic": { ko: "마이크 활동", en: "Mic Activity" },
  "signal.camera": { ko: "카메라", en: "Camera" },
  "signal.liveStress": { ko: "실시간 스트레스", en: "Live Stress" },
  "signal.feedbackState": { ko: "피드백 상태", en: "Feedback State" },
  "val.live": { ko: "라이브", en: "LIVE" },
  "val.done": { ko: "완료", en: "DONE" },
  "val.ready": { ko: "준비", en: "READY" },
  "val.on": { ko: "켜짐", en: "ON" },
  "val.preview": { ko: "미리보기", en: "PREVIEW" },
  "val.stby": { ko: "대기", en: "STBY" },
  "val.hidden": { ko: "숨김", en: "HIDDEN" },
  "state.live": { ko: "라이브 연습", en: "Live practice" },
  "state.ended": { ko: "세션 완료", en: "Session completed" },
  "state.starting": { ko: "시작 중", en: "Starting" },
  "state.error": { ko: "권한 필요", en: "Permission needed" },
  "state.ready": { ko: "시작 준비됨", en: "Ready to start" },

  // FeedbackEngine messages
  "fb.idle": { ko: "자세한 피드백은 세션 종료 후 제공됩니다.", en: "Detailed feedback appears after the session." },
  "fb.learning": { ko: "개인 기준선을 보정하는 중…", en: "Calibrating your personal baseline…" },
  "fb.stable": { ko: "완벽한 속도", en: "Perfect pace" },
  "fb.steady": { ko: "안정적인 속도 — 한 포인트씩 분명하게.", en: "Steady pace — let each point land." },
  "fb.mid": { ko: "조금 더 천천히 말해보세요", en: "Try to speak slower" },
  "fb.high": { ko: "깊게 숨을 들이쉬세요", en: "Take a deep breath" },
  "fb.activeQuiet": { ko: "조용히 듣는 중 — 뚜렷한 스트레스 급등 시에만 알림이 표시됩니다.", en: "Listening quietly — nudges appear on clear stress spikes." },

  // Signal note
  "note.simulated": {
    ko: "데모 신호 — Student 모델 추론 백엔드가 연결되기 전까지 stress와 confidence는 시뮬레이션입니다.",
    en: "Demo signal — stress & confidence are simulated until the Student-model inference backend is connected.",
  },
  "note.liveError": {
    ko: "분석 요청에 실패했거나 응답 시간이 초과됐습니다. 서버 실행 상태와 주소를 확인해 주세요.",
    en: "Analysis failed or timed out. Check the server status and URL.",
  },
  "note.liveOk": {
    ko: "모델 서버에 연결됨 · 실제 모델 분석 결과를 수신하고 있습니다.",
    en: "Connected to the model server · Receiving model analysis results.",
  },
  "note.liveVerified": {
    ko: "모델 서버 연결 확인 완료 · 마지막 분석 요청 기준",
    en: "Model connection verified on the last analysis request.",
  },
  "note.liveIdle": { ko: "연습을 시작하면 영상·음성을 수집해 분석합니다.", en: "Start practice to collect video and audio for analysis." },
  "note.preparation": { ko: "카메라·마이크로 분석 준비 데이터를 미리 수집합니다. 시작 전에는 브라우저에만 임시 보관하며, 연습을 시작하면 분석 서버로 전송합니다.", en: "We collect camera and microphone data for preparation. It stays temporarily in your browser until practice starts, when it is sent for analysis." },
  "note.preparationFrames": { ko: "영상 준비", en: "Video preparation" },
  "note.preparationAudioOn": { ko: "마이크 수집 중", en: "Collecting audio" },
  "note.preparationAudioOff": { ko: "마이크 준비 대기", en: "Audio preparation pending" },
  "note.prepareButton": { ko: "카메라·마이크 준비", en: "Prepare camera and microphone" },
  "note.liveCollecting": { ko: "첫 분석을 위한 영상·음성 수집 중 · 약 10초 후 분석 요청", en: "Collecting video and audio · First analysis requested after about 10 seconds" },
  "note.liveAnalyzing": { ko: "영상·음성 분석 중… 첫 결과는 시간이 걸릴 수 있습니다.", en: "Analyzing video and audio… The first result may take a while." },
  "note.liveCamera": { ko: "카메라 영상을 기다리는 중입니다. 카메라 권한과 미리보기를 확인해 주세요.", en: "Waiting for video. Check camera permission and preview." },
  "note.liveAudio": { ko: "마이크 데이터를 기다리는 중입니다. 마이크 권한을 확인해 주세요.", en: "Waiting for audio. Check microphone permission." },

  // Slides / deck
  "slide.hint": { ko: "← / → 키로 슬라이드 이동", en: "Use ← / → keys to move between slides" },
  "deck.addFile": { ko: "발표 파일을 추가하세요", en: "Add your presentation file" },
  "deck.addFileDesc": { ko: "PDF는 즉시 동작하거나, 샘플 프레젠테이션 덱을 사용하세요.", en: "PDF works instantly, or use the sample presentation deck." },
  "deck.pptDesc": { ko: "PowerPoint 미리보기는 PDF 변환이 필요합니다. 라이브 리허설 화면을 위해 PDF로 내보내세요.", en: "PowerPoint preview requires PDF conversion. Export this deck as PDF for the live rehearsal view." },

  // Report
  "report.timeline": { ko: "세션 스트레스 타임라인", en: "Session Stress Timeline" },
  "report.timelineSub": { ko: "세션 후 개요", en: "Post-session overview" },
  "report.diagnosis": { ko: "진단", en: "Diagnosis" },
  "report.diagnosisSub": { ko: "AI 분석 결과", en: "AI analysis result" },
  "report.overallJudgment": { ko: "종합 판단", en: "Overall Judgment" },
  "report.peakPoint": { ko: "최고 압박 지점", en: "Peak Pressure Point" },
  "report.summary": { ko: "세션 요약", en: "Session Summary" },
  "report.summarySub": { ko: "다음 리허설에서 개선할 점", en: "What to improve next rehearsal" },
  "report.overallAvg": { ko: "전체 평균 스트레스", en: "Overall Average Stress" },
  "report.overallAvgDesc": { ko: "순간 잡음을 제외한 세션 수준의 그림.", en: "Session-level picture without momentary noise." },
  "report.peakStress": { ko: "최고 스트레스 지점", en: "Peak Stress Point" },
  "report.peakStressDesc": { ko: "타임라인에서 가장 높은 단일 구간.", en: "Highest single bin across the timeline." },
  "report.mostStressfulPhase": { ko: "가장 스트레스 높은 단계", en: "Most Stressful Phase" },
  "report.mostStressfulPhaseDesc": { ko: "평균 압박이 가장 높았던 단계.", en: "Phase with the highest average pressure." },
  "report.nextStep": { ko: "추천 다음 단계", en: "Recommended Next Step" },
  "report.nextStepDesc": { ko: "다음 리허설의 집중 영역.", en: "Focus area for your next rehearsal." },
  "report.confidence": { ko: "Confidence", en: "Confidence" },
  "report.deck": { ko: "자료", en: "Deck" },
  "report.scene": { ko: "장면", en: "Scene" },
  "report.aiAgent": { ko: "AI 에이전트", en: "AI Agent" },
  "report.aiAgentSub": { ko: "후속 질문하기", en: "Ask follow-up questions" },
  "report.chatPlaceholder": { ko: "세션 결과에 대해 물어보세요...", en: "Ask about your session result..." },
  "report.ask": { ko: "질문", en: "Ask" },

  // Report dynamic text
  "phase.suffix": { ko: "단계", en: "phase" },
  "report.binLabel": { ko: "구간", en: "bin" },
  "report.stateStable": { ko: "대체로 안정적인 전달", en: "Mostly stable delivery" },
  "report.nextActionText": {
    ko: "주요 설명 블록 사이의 전환을 더 천천히 연습하고, 핵심 용어마다 짧게 쉬어 주세요.",
    en: "Practice slower transitions between your main explanation blocks and leave a short pause after each key term.",
  },
  "report.chatIntro": {
    ko: "세션이 끝나면 성과를 요약하고 후속 질문에 답해 드릴게요.",
    en: "After the session ends, I will summarize your performance and answer follow-up questions here.",
  },
  "report.chatWaiting": {
    ko: "세션이 끝나길 기다리는 중입니다. 종료하면 결과 해석을 도와드릴게요.",
    en: "Waiting for the session to finish. I will help interpret the results when you end it.",
  },
  "report.chatSummaryPrefix": { ko: "세션 요약이 준비됐어요. 가장 개선할 부분: ", en: "Session summary ready. Biggest improvement area: " },

  // AI agent replies
  "agent.stressPre": { ko: "가장 강한 압박 지점은 ", en: "The strongest pressure point appears around " },
  "agent.stressPost": { ko: " 부근으로 보입니다. 속도를 늦추고 핵심 용어 앞에 짧은 쉼을 넣어 보세요.", en: ". Focus on slowing your pace and adding a short pause before key terms." },
  "agent.stressFallback": { ko: "후반 중반부", en: "the later middle section" },
  "agent.calm": { ko: "꾸준한 속도가 가장 도움이 됩니다. 문장을 짧게 유지하고 한 포인트가 충분히 전달된 뒤 다음으로 넘어가세요.", en: "Steadier pacing will help most. Keep sentence lengths shorter and let one point land before moving to the next." },
  "agent.default": { ko: "먼저 압박이 높았던 설명 구간을 다시 보고, 기술적 내용으로 넘어가는 전환 하나를 더 차분하게 연습해 보세요.", en: "Review the high-pressure explanation segment first, then rehearse one calmer transition into your technical content." },

  // Agent starter pills
  "starter.0": { ko: "내 발표에서 가장 스트레스가 높은 부분은?", en: "Which part of my presentation seems most stressful?" },
  "starter.1": { ko: "다음 리허설 전에 무엇을 개선해야 할까?", en: "What should I improve before the next rehearsal?" },
  "starter.2": { ko: "기술 설명 중 더 차분하게 들리려면?", en: "How can I sound calmer during technical explanations?" },

  // Post-session highlights
  "highlight.0.title": { ko: "오프닝 안정성", en: "Opening Stability" },
  "highlight.0.detail": { ko: "첫 구간이 안정적으로 유지되어, 오프닝 스크립트가 이미 든든한 기준점이 되고 있습니다.", en: "The first segment stayed controlled, which means your opening script is already a strong anchor." },
  "highlight.1.title": { ko: "기술 설명 압박", en: "Technical Explanation Pressure" },
  "highlight.1.detail": { ko: "충분한 전환 표현 없이 세부 내용을 너무 빠르게 설명하면서 스트레스가 올라갔을 가능성이 있습니다.", en: "Stress likely rose while explaining detailed content too quickly without enough transition language." },
  "highlight.2.title": { ko: "최고의 개선 레버", en: "Best Improvement Lever" },
  "highlight.2.detail": { ko: "슬라이드 전환 사이의 짧은 쉼이 명료함과 자신감 모두를 높여줄 것입니다.", en: "Short pauses between slide changes will probably improve both clarity and perceived confidence." },

  // Session naming
  "save.title": { ko: "세션 저장", en: "Save Session" },
  "save.subtitle": { ko: "이 연습 세션의 이름을 정하세요", en: "Name this practice session" },
  "save.namePlaceholder": { ko: "세션 이름 입력...", en: "Enter a session name..." },
  "save.button": { ko: "세션 저장", en: "Save Session" },
  "save.saved": { ko: "저장됨", en: "Saved" },
  "save.defaultSuffix": { ko: "연습", en: "Practice" },

  // Errors / status
  "err.permission": { ko: "카메라 또는 마이크 권한이 거부되었습니다.", en: "Camera or microphone permission was denied." },
  "err.cameraPreview": { ko: "카메라 권한이 거부되었습니다. 시작 전에 본인 모습을 미리 보려면 접근을 허용하세요.", en: "Camera permission was denied. Allow access to preview yourself before starting." },
  "status.localOnly": { ko: "Supabase 미연결. 세션이 로컬에만 저장되었습니다.", en: "Supabase not connected. Session saved locally only." },
  "status.saved": { ko: "세션이 계정에 저장되었습니다.", en: "Session saved to your account." },
  "status.saveFailed": { ko: "세션 저장에 실패했습니다.", en: "Failed to save session." },

  // Scenes (by id)
  "scene.camera.label": { ko: "실시간 카메라", en: "Live Camera" },
  "scene.camera.desc": { ko: "연습 중 실제 웹캠 미리보기를 사용합니다.", en: "Use your real webcam preview during practice." },
  "scene.audience.label": { ko: "강의실 청중", en: "Classroom Audience" },
  "scene.audience.desc": { ko: "강의실 청중이 당신을 바라보는 환경에서 연습합니다.", en: "Practice with a classroom audience looking toward you." },
  "scene.interview.label": { ko: "면접 패널", en: "Interview Panel" },
  "scene.interview.desc": { ko: "차분한 모의 면접 패널과 함께 연습합니다.", en: "Practice with a calm mock interview panel." },
  "scene.slides.label": { ko: "발표 슬라이드", en: "Presentation Slides" },
  "scene.slides.desc": { ko: "업로드한 자료를 메인 리허설 화면으로 표시합니다.", en: "Show your uploaded deck as the main rehearsal screen." },
  "scene.classroom-slides.label": { ko: "강의실 속 슬라이드", en: "Slides in Classroom" },
  "scene.classroom-slides.desc": { ko: "강의실 청중 앞에서 자료를 띄우고 연습합니다.", en: "Practice with your deck in front of a classroom audience." },
  "scene.audienceCaption": { ko: "강의실 청중 리허설", en: "Classroom audience rehearsal" },
  "interview.dr.name": { ko: "연구 책임자", en: "Research Lead" },
  "interview.dr.q": { ko: "모델 선택 이유를 1분 안에 설명해 주세요.", en: "Explain your model choice in one minute." },
  "interview.pm.name": { ko: "제품 리뷰어", en: "Product Reviewer" },
  "interview.pm.q": { ko: "실제 교실에서는 어떻게 동작하나요?", en: "How would this work in a real classroom?" },
  "interview.ai.name": { ko: "기술 패널", en: "Technical Panel" },
  "interview.ai.q": { ko: "카메라 각도가 바뀌면 어떻게 되나요?", en: "What happens when the camera angle changes?" },
};

type LangContextValue = {
  lang: Lang;
  setLang: (l: Lang) => void;
  toggle: () => void;
  t: (key: string, fallback?: string) => string;
};

const LangContext = createContext<LangContextValue | null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(DEFAULT_LANG);

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "ko" || stored === "en") setLangState(stored);
  }, []);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    window.localStorage.setItem(STORAGE_KEY, l);
    document.documentElement.lang = l;
  }, []);

  const toggle = useCallback(() => setLang(lang === "ko" ? "en" : "ko"), [lang, setLang]);

  const t = useCallback(
    (key: string, fallback?: string) => dict[key]?.[lang] ?? dict[key]?.en ?? fallback ?? key,
    [lang],
  );

  const value = useMemo(() => ({ lang, setLang, toggle, t }), [lang, setLang, toggle, t]);

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useLang must be used within LanguageProvider.");
  return ctx;
}
