"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BrainCircuit, Languages, LockKeyhole, Mail, Sparkles, UserRound, Video } from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { useLang } from "@/lib/i18n";
import { getSupabaseBrowserClient } from "@/lib/supabase/client";

export function AuthPanel() {
  const [mode, setMode] = useState<"login" | "signup" | "forgot">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [submitMessage, setSubmitMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();
  const { hasEnv, signInWithPassword, signUpWithPassword, user } = useSupabaseAuth();
  const { lang, toggle, t } = useLang();

  useEffect(() => {
    if (user) router.push("/dashboard");
  }, [router, user]);

  async function handleForgotPassword() {
    try {
      setIsSubmitting(true);
      setSubmitError("");
      setSubmitMessage("");
      const supabase = getSupabaseBrowserClient();
      if (!supabase) throw new Error("Supabase not configured.");
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/callback`,
      });
      if (error) throw error;
      setSubmitMessage(t("auth.recoverySent"));
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : t("auth.recoveryFailed"));
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleSubmit() {
    try {
      setIsSubmitting(true);
      setSubmitError("");
      setSubmitMessage("");

      if (mode === "signup") {
        const result = await signUpWithPassword(email, password, name);
        if (result.needsEmailConfirmation) {
          setSubmitMessage(t("auth.created"));
          return;
        }
      } else {
        await signInWithPassword(email, password);
      }

      router.push("/dashboard");
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : t("auth.failed"));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="auth-layout">
      <button
        type="button"
        className="lang-toggle auth-lang-toggle"
        onClick={toggle}
        aria-label={t("lang.toggle")}
        title={t("lang.toggle")}
      >
        <Languages size={14} />
        <span>{lang === "ko" ? "한" : "EN"}</span>
      </button>

      <div className="auth-left">
        <div className="auth-brand-row">
          <div className="auth-brand-icon">
            <Sparkles size={18} color="white" />
          </div>
          <div>
            <div className="auth-brand-name">PreSense</div>
            <div className="auth-brand-sub">{t("brand.tagline")}</div>
          </div>
        </div>

        <h1 className="auth-headline">{t("auth.headline")}</h1>
        <p className="auth-desc">{t("auth.desc")}</p>

        <div className="auth-features">
          <div className="auth-feature">
            <div className="auth-feature-icon"><Video size={15} color="white" /></div>
            {t("auth.feature1")}
          </div>
          <div className="auth-feature">
            <div className="auth-feature-icon"><BrainCircuit size={15} color="white" /></div>
            {t("auth.feature2")}
          </div>
          <div className="auth-feature">
            <div className="auth-feature-icon"><UserRound size={15} color="white" /></div>
            {t("auth.feature3")}
          </div>
        </div>
      </div>

      <div className="auth-right">
        <div className="auth-form-card">
          <div className="auth-form-title">{mode === "login" ? t("auth.welcomeBack") : t("auth.createAccount")}</div>
          <div className="auth-form-sub">
            {mode === "login" ? t("auth.signInSub") : t("auth.signUpSub")}
          </div>

          <div className="auth-tabs-row">
            <button type="button" className={`auth-tab${mode === "login" ? " active" : ""}`} onClick={() => { setMode("login"); setSubmitError(""); setSubmitMessage(""); }}>
              {t("auth.login")}
            </button>
            <button type="button" className={`auth-tab${mode === "signup" ? " active" : ""}`} onClick={() => { setMode("signup"); setSubmitError(""); setSubmitMessage(""); }}>
              {t("auth.signUp")}
            </button>
          </div>

          {mode === "forgot" ? (
            <>
              <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "14px 0 16px" }}>
                {t("auth.forgotIntro")}
              </p>
              <div className="field-group">
                <div className="field-wrap">
                  <label className="field-label">{t("auth.email")}</label>
                  <div className="field-input">
                    <Mail size={15} className="field-icon" />
                    <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com" type="email" />
                  </div>
                </div>
              </div>
              {submitError && <p className="error-msg" style={{ marginTop: 10 }}>{submitError}</p>}
              {submitMessage && <p className="success-msg" style={{ marginTop: 10 }}>{submitMessage}</p>}
              <button
                type="button"
                className="btn btn-primary"
                style={{ width: "100%", justifyContent: "center", marginTop: 12 }}
                disabled={!hasEnv || isSubmitting || !email}
                onClick={handleForgotPassword}
              >
                {isSubmitting ? t("auth.sending") : t("auth.sendRecovery")}
              </button>
              <button
                type="button"
                style={{ width: "100%", marginTop: 10, background: "none", border: "none", cursor: "pointer", fontSize: 13, color: "var(--text-secondary)" }}
                onClick={() => { setMode("login"); setSubmitError(""); setSubmitMessage(""); }}
              >
                {t("auth.backToLogin")}
              </button>
            </>
          ) : (
            <>
              <div className="field-group">
                {mode === "signup" && (
                  <div className="field-wrap">
                    <label className="field-label">{t("auth.name")}</label>
                    <div className="field-input">
                      <UserRound size={15} className="field-icon" />
                      <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t("auth.namePlaceholder")} />
                    </div>
                  </div>
                )}

                <div className="field-wrap">
                  <label className="field-label">{t("auth.email")}</label>
                  <div className="field-input">
                    <Mail size={15} className="field-icon" />
                    <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com" type="email" />
                  </div>
                </div>

                <div className="field-wrap">
                  <label className="field-label">{t("auth.password")}</label>
                  <div className="field-input">
                    <LockKeyhole size={15} className="field-icon" />
                    <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder={t("auth.passwordPlaceholder")} />
                  </div>
                </div>
              </div>

              {mode === "login" && (
                <button
                  type="button"
                  style={{ background: "none", border: "none", cursor: "pointer", fontSize: 12, color: "var(--text-secondary)", padding: "4px 0", marginTop: 2, textAlign: "right", width: "100%" }}
                  onClick={() => { setMode("forgot"); setSubmitError(""); setSubmitMessage(""); }}
                >
                  {t("auth.forgotPassword")}
                </button>
              )}

              {!hasEnv && (
                <p className="info-msg">{t("auth.notConfigured")}</p>
              )}
              {submitError && <p className="error-msg">{submitError}</p>}
              {submitMessage && <p className="success-msg">{submitMessage}</p>}

              <button
                type="button"
                className="btn btn-primary"
                style={{ width: "100%", justifyContent: "center", marginTop: 4 }}
                disabled={!hasEnv || isSubmitting || !email || !password || (mode === "signup" && !name)}
                onClick={handleSubmit}
              >
                {isSubmitting ? t("auth.pleaseWait") : mode === "login" ? t("auth.enterDashboard") : t("auth.createAccountBtn")}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
