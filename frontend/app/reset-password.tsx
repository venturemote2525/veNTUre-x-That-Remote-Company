import { useEffect } from "react";
import { useRouter } from "expo-router";
import { useURL } from "expo-linking";
import { supabase } from "@/lib/supabase";
import { ThemedSafeAreaView, Text } from "@/components/Themed";

export default function ResetPasswordHandler() {
  const router = useRouter();
  const url = useURL();

  useEffect(() => {
    console.log("Deep link URL received:", url);

    if (!url) {
      console.log("No URL received — waiting for deep link");
      return;
    }

    try {
      const [path, hash] = url.split("#");
      console.log("Path:", path, "Hash:", hash);

      const hashParams = new URLSearchParams(hash || "");
      const type = hashParams.get("type");
      const access_token = hashParams.get("access_token");
      const refresh_token = hashParams.get("refresh_token");

      console.log("Parsed reset params:", { type, access_token, refresh_token });

      if (type === "recovery" && access_token && refresh_token) {
        supabase.auth
          .setSession({ access_token, refresh_token })
          .then(({ error }) => {
            if (error) {
              console.error("Error setting session:", error);
              router.replace("/(auth)/login");
            } else {
              console.log("Session set — routing to new-password");
              router.replace("/(auth)/new-password");
            }
          });
      } else {
        console.log("Invalid params — routing to login");
        router.replace("/(auth)/login");
      }
    } catch (error) {
      console.error("Error parsing reset URL:", error);
      router.replace("/(auth)/login");
    }
  }, [url]);

  return (
    <ThemedSafeAreaView className="flex-1 justify-center items-center">
      <Text>Processing password reset...</Text>
    </ThemedSafeAreaView>
  );
}
