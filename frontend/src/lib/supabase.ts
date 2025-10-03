// src/lib/supabase.ts

import { createClient } from "@supabase/supabase-js";

// ⚠️ Use import.meta.env for Vite environment variables
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  // Use console.error instead of throw new Error to avoid completely crashing the app silently
  console.error(
    "FATAL ERROR: Missing Supabase environment variables. Did you set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env.local?",
  );
  // If you must prevent the app from continuing, you need an Error Boundary,
  // but for debugging, let's return a null client or similar to see an error on screen later.
  // For now, keep the throw, but ensure the variables are being read correctly.
  throw new Error("Missing Supabase environment variables.");
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
