import { createClient } from "@supabase/supabase-js";

// Ensure these environment variables are available in your React app (e.g., in a .env.local file)
// NOTE: For the frontend, you should use the *public* Supabase URL and the *anon* key.
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    "Missing Supabase environment variables. Make sure NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY are set.",
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
