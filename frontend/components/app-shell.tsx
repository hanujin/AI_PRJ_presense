"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { BarChart3, Bell, HelpCircle, LogOut, Search, Settings, Sparkles, UserRound, Video } from "lucide-react";
import { useSupabaseAuth } from "@/components/supabase-provider";

const navLinks = [
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/practice", label: "Practice", icon: Video },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { hasEnv, isLoading, signOut, user } = useSupabaseAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isLoading && hasEnv && !user) router.push("/");
  }, [hasEnv, isLoading, router, user]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const displayName = user?.user_metadata?.full_name ?? user?.email ?? "Demo User";
  const initials = displayName.slice(0, 2).toUpperCase();

  async function handleSignOut() {
    setMenuOpen(false);
    await signOut();
    router.push("/");
  }

  if (hasEnv && isLoading) {
    return (
      <div className="layout">
        <aside className="sidebar">
          <Link href="/" className="sidebar-brand">
            <div className="sidebar-brand-icon"><Sparkles size={16} /></div>
            <div>
              <strong>PreSense</strong>
              <span>Presentation AI</span>
            </div>
          </Link>
        </aside>
        <div className="main-area">
          <div className="page-content">
            <p className="state-loading">Loading your workspace...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <Link href="/" className="sidebar-brand">
          <div className="sidebar-brand-icon"><Sparkles size={16} /></div>
          <div>
            <strong>PreSense</strong>
            <span>Presentation AI</span>
          </div>
        </Link>

        <nav className="sidebar-nav">
          {navLinks.map(({ href, label, icon: Icon }) => (
            <Link key={href} href={href} className={`sidebar-link${pathname === href ? " active" : ""}`}>
              <Icon size={16} />
              {label}
            </Link>
          ))}
        </nav>

        <div className="sidebar-footer">
          <Link href="/practice" className="sidebar-cta">
            <Video size={15} />
            Start Practice
          </Link>
        </div>
      </aside>

      <div className="main-area">
        <header className="main-header">
          <div className="search-bar">
            <Search size={14} />
            <input placeholder="Search sessions..." />
          </div>

          <div className="header-actions">
            <button type="button" className="icon-btn" aria-label="Notifications">
              <Bell size={15} />
            </button>
            <button type="button" className="icon-btn" aria-label="Help">
              <HelpCircle size={15} />
            </button>

            <div style={{ position: "relative" }} ref={menuRef}>
              <button
                type="button"
                className="avatar-btn"
                onClick={() => setMenuOpen((v) => !v)}
                aria-label="Account menu"
              >
                {initials}
              </button>

              {menuOpen && (
                <div className="avatar-menu">
                  <div className="avatar-menu-label">{displayName}</div>
                  <div className="avatar-menu-divider" />
                  <Link href="/settings" className="avatar-menu-item" onClick={() => setMenuOpen(false)}>
                    <Settings size={14} />
                    Settings
                  </Link>
                  <div className="avatar-menu-divider" />
                  {user ? (
                    <button type="button" className="avatar-menu-item danger" onClick={handleSignOut}>
                      <LogOut size={14} />
                      Sign out
                    </button>
                  ) : (
                    <Link href="/" className="avatar-menu-item" onClick={() => setMenuOpen(false)}>
                      <UserRound size={14} />
                      Log in
                    </Link>
                  )}
                </div>
              )}
            </div>
          </div>
        </header>

        {children}
      </div>
    </div>
  );
}
