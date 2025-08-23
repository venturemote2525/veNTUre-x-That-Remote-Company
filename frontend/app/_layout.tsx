import '@/app/globals.css';
import { GluestackUIProvider } from '@/components/ui/gluestack-ui-provider';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect } from 'react';
import 'react-native-reanimated';

export default function RootLayout() {
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
    'Poppins-Bold': require('@/assets/fonts/Poppins-Bold.ttf'),
    'Poppins-SemiBold': require('@/assets/fonts/Poppins-SemiBold.ttf'),
    'Poppins-Medium': require('@/assets/fonts/Poppins-Medium.ttf'),
    'Poppins-Regular': require('@/assets/fonts/Poppins-Regular.ttf'),
  });

  if (!loaded) {
    // Async font loading only occurs in development.
    return null;
  }

  return (
<<<<<<< Updated upstream
    <GluestackUIProvider mode="light">
      <Stack>
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="(auth)" options={{ headerShown: false }} />
        <Stack.Screen name="(logging)" options={{ headerShown: false }} />
        <Stack.Screen name="(user)" options={{ headerShown: false }} />
        <Stack.Screen name="+not-found" options={{ headerShown: false }} />
      </Stack>
      <StatusBar style="auto" />
    </GluestackUIProvider>
=======
    <AuthProvider>
      <GluestackUIProvider mode="light">
        <RootLayoutNav />
      </GluestackUIProvider>
    </AuthProvider>
  );
}

function RootLayoutNav() {
  const { profile, authenticated, loading, profileLoading } = useAuth();
  const router = useRouter();
  const segments = useSegments();

  useEffect(() => {
    // wait until loading finishes
    if (!loading && !profileLoading) {
      const inAuthScreens = segments[0] === '(auth)';
      console.log('authenticated:', authenticated);
      // Navigate to onboarding if incomplete
      if (authenticated && profile === null && !inAuthScreens) {
        router.replace('/(auth)/onboarding');
      }
      // Go to root if not authenticated
      if (!authenticated && !inAuthScreens) {
        router.replace('/');
      }
    }
  }, [loading, profileLoading, authenticated, profile, router, segments]);

  return (
    <Stack>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="(auth)" options={{ headerShown: false }} />
      <Stack.Screen name="(logging)" options={{ headerShown: false }} />
      <Stack.Screen name="(user)" options={{ headerShown: false }} />
      <Stack.Screen name="+not-found" options={{ headerShown: false }} />
    </Stack>
>>>>>>> Stashed changes
  );
}
