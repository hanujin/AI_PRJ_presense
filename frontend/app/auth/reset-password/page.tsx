"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { LockKeyhole, Sparkles } from "lucide-react";
import { getSupabaseBrowserClient } from "@/lib/supabase/client";

export default function ResetPasswordPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleReset() {
    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    try {
      setIsSubmitting(true);
      setError("");
      const supabase = getSupabaseBrowserClient();
      if (!supabase) throw new Error("Supabase not configured.");

      const { error } = await supabase.auth.updateUser({ password });
      if (error) throw error;

      setMessage("Password updated successfully. Redirecting…");
      setTimeout(() => router.push("/dashboard"), 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update password.");
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
        <h1 className="auth-headline">Set a new password.</h1>
        <p className="auth-desc">Choose a strong password to protect your PreSense account.</p>
      </div>

      <div className="auth-right">
        <div className="auth-form-card">
          <div className="auth-form-title">Reset password</div>
          <div className="auth-form-sub">Enter and confirm your new password below.</div>

          <div className="field-group" style={{ marginTop: 20 }}>
            <div className="field-wrap">
              <label className="field-label">New Password</label>
              <div className="field-input">
                <LockKeyhole size={15} className="field-icon" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 6 characters"
                />
              </div>
            </div>

            <div className="field-wrap">
              <label className="field-label">Confirm Password</label>
              <div className="field-input">
                <LockKeyhole size={15} className="field-icon" />
                <input
                  type="password"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="Re-enter your password"
                  onKeyDown={(e) => { if (e.key === "Enter") handleReset(); }}
                />
              </div>
            </div>
          </div>

          {error && <p className="error-msg" style={{ marginTop: 10 }}>{error}</p>}
          {message && <p className="success-msg" style={{ marginTop: 10 }}>{message}</p>}

          <button
            type="button"
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", marginTop: 16 }}
            disabled={isSubmitting || !password || !confirm}
            onClick={handleReset}
          >
            {isSubmitting ? "Updating…" : "Update Password"}
          </button>
        </div>
      </div>
    </div>
  );
}
