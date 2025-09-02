import '@/app/globals.css';
import { GluestackUIProvider } from '@/components/ui/gluestack-ui-provider';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect, useState } from 'react';
import 'react-native-reanimated';
import { NativeModules, NativeEventEmitter } from 'react-native';
import { ICDeviceProvider } from '@/context/ICDeviceContext';

const { ICDeviceModule } = NativeModules;

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
    <ICDeviceProvider>
      <AuthProvider>
        <GluestackUIProvider mode="light">
          <RootLayoutNav />
        </GluestackUIProvider>
      </AuthProvider>
    </ICDeviceProvider>
  );
}

function RootLayoutNav() {
  const { profile, authenticated, loading, profileLoading } = useAuth();
  const router = useRouter();
  const segments = useSegments();
  const [init, setInit] = useState(false);

  useEffect(() => {
    // wait until loading finishes
    if (!loading && !profileLoading) {
      const inAuthScreens = segments[0] === '(auth)';

      // Navigate to onboarding if authenticated but profile not set
      if (authenticated && profile === null && !inAuthScreens) {
        router.replace('/(auth)/onboarding');
      }
      // Navigate to tabs if fully authenticated
      else if (authenticated && profile !== null && segments[0] !== '(tabs)' && !init) {
        setInit(true);
        router.replace('/(tabs)/home');
      }
      // Navigate to root if not authenticated
      else if (!authenticated && !inAuthScreens) {
        router.replace('/');
      }
    }
  }, [loading, profileLoading, authenticated, profile, router, segments, init]);

  return (
    <Stack>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="(auth)" options={{ headerShown: false }} />
      <Stack.Screen name="(logging)" options={{ headerShown: false }} />
      <Stack.Screen name="(user)" options={{ headerShown: false }} />
      <Stack.Screen name="+not-found" options={{ headerShown: false }} />
    </Stack>
  );
}
