-- Run in the Supabase SQL editor if video upload reports a missing bucket,
-- permission, or column. This migration preserves existing session records.
begin;

alter table public.presentation_records
  add column if not exists timeline jsonb not null default '[]'::jsonb;
alter table public.presentation_records
  add column if not exists video_path text;

insert into storage.buckets (id, name, public)
values ('session-videos', 'session-videos', false)
on conflict (id) do nothing;

do $$
begin
  if not exists (
    select 1 from pg_policies where schemaname = 'storage'
      and tablename = 'objects'
      and policyname = 'Users can manage their own session videos'
  ) then
    create policy "Users can manage their own session videos"
    on storage.objects for all to authenticated
    using (bucket_id = 'session-videos' and (storage.foldername(name))[1] = (select auth.uid()::text))
    with check (bucket_id = 'session-videos' and (storage.foldername(name))[1] = (select auth.uid()::text));
  end if;
end $$;

commit;
