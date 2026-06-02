import { AppShell } from "@/components/app-shell";
import { DashboardShell } from "@/components/dashboard-shell";

export default function PracticePage() {
  return (
    <AppShell>
      <DashboardShell />
    </AppShell>
  );
}
