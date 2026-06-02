"use client";

import { useEffect, useState } from "react";
import { Bell, Camera, Lock, Mic, SlidersHorizontal } from "lucide-react";
import {
  buildDefaultPrimarySettings,
  buildDefaultSectionSettings,
  primarySettings,
  settingsSections,
} from "@/lib/app-data";
import { useSupabaseAuth } from "@/components/supabase-provider";
import { loadUserSettings, saveUserSettings } from "@/lib/supabase/data";

const PRIMARY_SETTINGS_KEY = "presense-primary-settings";
const SECTION_SETTINGS_KEY = "presense-section-settings";

const PRIMARY_ICONS = [Camera, Mic, Bell];

export function SettingsPanel() {
  const { hasEnv, user } = useSupabaseAuth();
  const [primaryToggles, setPrimaryToggles] = useState<Record<string, boolean>>(() => buildDefaultPrimarySettings());
  const [sectionToggles, setSectionToggles] = useState<Record<string, boolean>>(() => buildDefaultSectionSettings());
  const [statusMessage, setStatusMessage] = useState("");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    if (!hasEnv || !user) { setLoadError(""); return; }
    let isMounted = true;

    void loadUserSettings()
      .then((settings) => {
        if (!isMounted) return;
        setPrimaryToggles(settings.primary);
        setSectionToggles(settings.sections);
        setLoadError("");
      })
      .catch((error) => {
        if (!isMounted) return;
        setLoadError(error instanceof Error ? error.message : "Failed to load settings.");
      });

    return () => { isMounted = false; };
  }, [hasEnv, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    const p = window.localStorage.getItem(PRIMARY_SETTINGS_KEY);
    const s = window.localStorage.getItem(SECTION_SETTINGS_KEY);
    if (p) setPrimaryToggles((c) => ({ ...c, ...JSON.parse(p) }));
    if (s) setSectionToggles((c) => ({ ...c, ...JSON.parse(s) }));
  }, [hasEnv, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    window.localStorage.setItem(PRIMARY_SETTINGS_KEY, JSON.stringify(primaryToggles));
  }, [hasEnv, primaryToggles, user]);

  useEffect(() => {
    if (hasEnv && user) return;
    window.localStorage.setItem(SECTION_SETTINGS_KEY, JSON.stringify(sectionToggles));
  }, [hasEnv, sectionToggles, user]);

  async function persistSettings(nextP: Record<string, boolean>, nextS: Record<string, boolean>) {
    if (!hasEnv || !user) { setStatusMessage("Saved in this browser only."); return; }
    try {
      await saveUserSettings({ primary: nextP, sections: nextS });
      setStatusMessage("Saved to your account.");
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Failed to save settings.");
    }
  }

  function togglePrimary(id: string) {
    const next = { ...primaryToggles, [id]: !primaryToggles[id] };
    setPrimaryToggles(next);
    void persistSettings(next, sectionToggles);
  }

  function toggleSection(id: string) {
    const next = { ...sectionToggles, [id]: !sectionToggles[id] };
    setSectionToggles(next);
    void persistSettings(primaryToggles, next);
  }

  const SECTION_ICONS = [SlidersHorizontal, Lock, Bell];

  return (
    <main className="page-content">
      <div className="page-top">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">Manage your practice preferences, privacy, and AI coaching options.</p>
        </div>
      </div>

      {loadError && <p className="error-msg" style={{ marginBottom: 16 }}>{loadError}</p>}
      {statusMessage && <p className="success-msg" style={{ marginBottom: 16 }}>{statusMessage}</p>}

      {/* Primary toggles */}
      <div className="kpi-row" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        {primarySettings.map((setting, i) => {
          const Icon = PRIMARY_ICONS[i] ?? Camera;
          const isOn = primaryToggles[setting.id];
          return (
            <div key={setting.id} className="kpi-card">
              <div className="kpi-top">
                <div className="kpi-icon" style={isOn ? { background: "var(--accent-ok-light)", color: "var(--accent-ok)" } : {}}>
                  <Icon size={16} />
                </div>
                <button
                  type="button"
                  className={`toggle-btn${isOn ? " on" : ""}`}
                  aria-pressed={isOn}
                  onClick={() => togglePrimary(setting.id)}
                >
                  {isOn ? "Enabled" : "Disabled"}
                </button>
              </div>
              <div className="kpi-value" style={{ fontSize: 15, fontFamily: "inherit", marginBottom: 4 }}>{setting.label}</div>
              <div className="kpi-sub">Click to toggle</div>
            </div>
          );
        })}
      </div>

      {/* Section settings */}
      <div className="settings-grid">
        {settingsSections.map((section, i) => {
          const Icon = SECTION_ICONS[i] ?? SlidersHorizontal;
          return (
            <div key={section.title} className="card">
              <div className="panel-head">
                <div className="panel-head-left">
                  <span className="panel-title">{section.title}</span>
                </div>
                <Icon size={16} color="var(--text-secondary)" />
              </div>

              {section.items.map((item) => {
                const isOn = sectionToggles[item.id];
                return (
                  <div key={item.id} className="settings-row">
                    <div>
                      <div className="settings-row-label">{item.label}</div>
                    </div>
                    <button
                      type="button"
                      className={`toggle-btn${isOn ? " on" : ""}`}
                      aria-pressed={isOn}
                      onClick={() => toggleSection(item.id)}
                    >
                      {isOn ? "On" : "Off"}
                    </button>
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </main>
  );
}
