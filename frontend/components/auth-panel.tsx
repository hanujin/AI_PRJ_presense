"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { BrainCircuit, LockKeyhole, Mail, Sparkles, UserRound, Video } from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";
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
      setSubmitMessage("Recovery email sent. Check your inbox and click the link.");
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to send recovery email.");
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
          setSubmitMessage("Account created. Check your email to confirm before logging in.");
          return;
        }
      } else {
        await signInWithPassword(email, password);
      }

      router.push("/dashboard");
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Authentication failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="auth-layout">
      <div className="auth-left">
        <div className="auth-brand-row">
          <div className="auth-brand-icon">
            <Sparkles size={18} color="white" />
          </div>
          <div>
            <div className="auth-brand-name">PreSense</div>
            <div className="auth-brand-sub">Presentation AI</div>
          </div>
        </div>

        <h1 className="auth-headline">Practice smarter, present better.</h1>
        <p className="auth-desc">
          Real-time stress detection from your webcam and microphone — powered by Knowledge
          Distillation from biosignals. No wearables required.
        </p>

        <div className="auth-features">
          <div className="auth-feature">
            <div className="auth-feature-icon"><Video size={15} color="white" /></div>
            Live camera + microphone analysis
          </div>
          <div className="auth-feature">
            <div className="auth-feature-icon"><BrainCircuit size={15} color="white" /></div>
            AI stress detection after each session
          </div>
          <div className="auth-feature">
            <div className="auth-feature-icon"><UserRound size={15} color="white" /></div>
            Session history and progress tracking
          </div>
        </div>
      </div>

      <div className="auth-right">
        <div className="auth-form-card">
          <div className="auth-form-title">{mode === "login" ? "Welcome back" : "Create account"}</div>
          <div className="auth-form-sub">
            {mode === "login" ? "Sign in to your PreSense account." : "Start your presentation practice journey."}
          </div>

          <div className="auth-tabs-row">
            <button type="button" className={`auth-tab${mode === "login" ? " active" : ""}`} onClick={() => { setMode("login"); setSubmitError(""); setSubmitMessage(""); }}>
              Login
            </button>
            <button type="button" className={`auth-tab${mode === "signup" ? " active" : ""}`} onClick={() => { setMode("signup"); setSubmitError(""); setSubmitMessage(""); }}>
              Sign Up
            </button>
          </div>

          {mode === "forgot" ? (
            <>
              <p style={{ fontSize: 13, color: "var(--text-secondary)", margin: "14px 0 16px" }}>
                Enter your account email and we will send a password recovery link.
              </p>
              <div className="field-group">
                <div className="field-wrap">
                  <label className="field-label">Email</label>
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
                {isSubmitting ? "Sending…" : "Send Recovery Email"}
              </button>
              <button
                type="button"
                style={{ width: "100%", marginTop: 10, background: "none", border: "none", cursor: "pointer", fontSize: 13, color: "var(--text-secondary)" }}
                onClick={() => { setMode("login"); setSubmitError(""); setSubmitMessage(""); }}
              >
                ← Back to login
              </button>
            </>
          ) : (
            <>
              <div className="field-group">
                {mode === "signup" && (
                  <div className="field-wrap">
                    <label className="field-label">Name</label>
                    <div className="field-input">
                      <UserRound size={15} className="field-icon" />
                      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Your full name" />
                    </div>
                  </div>
                )}

                <div className="field-wrap">
                  <label className="field-label">Email</label>
                  <div className="field-input">
                    <Mail size={15} className="field-icon" />
                    <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="name@example.com" type="email" />
                  </div>
                </div>

                <div className="field-wrap">
                  <label className="field-label">Password</label>
                  <div className="field-input">
                    <LockKeyhole size={15} className="field-icon" />
                    <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Enter password" />
                  </div>
                </div>
              </div>

              {mode === "login" && (
                <button
                  type="button"
                  style={{ background: "none", border: "none", cursor: "pointer", fontSize: 12, color: "var(--text-secondary)", padding: "4px 0", marginTop: 2, textAlign: "right", width: "100%" }}
                  onClick={() => { setMode("forgot"); setSubmitError(""); setSubmitMessage(""); }}
                >
                  Forgot password?
                </button>
              )}

              {!hasEnv && (
                <p className="info-msg">Supabase not configured — auth is disabled.</p>
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
                {isSubmitting ? "Please wait..." : mode === "login" ? "Enter Dashboard" : "Create Account"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
