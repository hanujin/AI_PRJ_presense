create extension if not exists pgcrypto;

create table if not exists public.profiles (
  id uuid primary key references auth.users on delete cascade,
  email text,
  full_name text,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.presentation_records (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  title text not null,
  session_date timestamptz not null default timezone('utc', now()),
  duration_seconds integer not null default 0,
  result text not null,
  stress_average numeric(4, 2) not null,
  diagnosis text not null,
  next_action text not null default '',
  scene_label text not null default '',
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.diagnosis_history (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users on delete cascade,
  label text not null,
  value text not null,
  detail text not null,
  created_at timestamptz not null default timezone('utc', now())
);

create table if not exists public.user_settings (
  user_id uuid primary key references auth.users on delete cascade,
  primary_settings jsonb not null default '{}'::jsonb,
  section_settings jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);

grant select, insert, update, delete on public.profiles to authenticated;
grant select, insert, update, delete on public.presentation_records to authenticated;
grant select, insert, update, delete on public.diagnosis_history to authenticated;
grant select, insert, update, delete on public.user_settings to authenticated;

alter table public.profiles enable row level security;
alter table public.presentation_records enable row level security;
alter table public.diagnosis_history enable row level security;
alter table public.user_settings enable row level security;

create policy "Users can manage their own profile"
on public.profiles
for all
using ((select auth.uid()) = id)
with check ((select auth.uid()) = id);

create policy "Users can manage their own presentation records"
on public.presentation_records
for all
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "Users can manage their own diagnosis history"
on public.diagnosis_history
for all
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "Users can manage their own settings"
on public.user_settings
for all
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create or replace function public.handle_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc', now());
  return new;
end;
$$;

drop trigger if exists user_settings_set_updated_at on public.user_settings;

create trigger user_settings_set_updated_at
before update on public.user_settings
for each row execute procedure public.handle_updated_at();
