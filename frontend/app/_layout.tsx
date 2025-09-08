import '@/app/globals.css';
import { GluestackUIProvider } from '@/components/ui/gluestack-ui-provider';
import { AuthProvider, useAuth } from '@/context/AuthContext';
import { useFonts } from 'expo-font';
import { Stack, useRouter, useSegments } from 'expo-router';
import { useEffect, useState } from 'react';
import 'react-native-reanimated';
import { ICDeviceProvider, useICDevice } from '@/context/ICDeviceContext';
import { ThemeProvider, useThemeMode } from '@/context/ThemeContext';

export default function RootLayout() {
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
    'Fredoka-Bold': require('@/assets/fonts/Fredoka-Bold.ttf'),
    'Fredoka-SemiBold': require('@/assets/fonts/Fredoka-SemiBold.ttf'),
    'Fredoka-Medium': require('@/assets/fonts/Fredoka-Medium.ttf'),
    'Fredoka-Regular': require('@/assets/fonts/Fredoka-Regular.ttf'),
  });

  if (!loaded) {
    // Async font loading only occurs in development.
    return null;
  }

  return (
    <AuthProvider>
      <ICDeviceProvider>
        <ThemeProvider>
          <RootLayoutNav />
        </ThemeProvider>
      </ICDeviceProvider>
    </AuthProvider>
  );
}

function RootLayoutNav() {
  const { mode } = useThemeMode();
  const { profile, authenticated, loading, profileLoading } = useAuth();
  const { connectedDevices, getLatestWeightForDevice, weightData } =
    useICDevice();
  const router = useRouter();
  const segments = useSegments();
  const [init, setInit] = useState(false);

  // -------------------- Auth Routing --------------------
  useEffect(() => {
    // wait until loading finishes
    if (!loading && !profileLoading) {
      const inAuthScreens = segments[0] === '(auth)';

      // Navigate to onboarding if authenticated but profile not set
      if (authenticated && profile === null && !inAuthScreens) {
        router.replace('/(auth)/onboarding');
      }
      // Navigate to tabs if fully authenticated
      else if (
        authenticated &&
        profile !== null &&
        segments[0] !== '(tabs)' &&
        !init
      ) {
        setInit(true);
        router.replace('/(tabs)/home');
      }
      // Navigate to root if not authenticated
      else if (!authenticated && !inAuthScreens) {
        router.replace('/');
      }
    }
  }, [loading, profileLoading, authenticated, profile, router, segments, init]);

  // -------------------- Weight taking --------------------
  const [lastRoutedWeightTs, setLastRoutedWeightTs] = useState<number | null>(
    null,
  );
  useEffect(() => {
    console.log('connected', connectedDevices);
    if (connectedDevices.length === 0) return;

    const deviceWithNewWeight = connectedDevices.find(device => {
      const latestWeight = getLatestWeightForDevice(device.mac);
      if (!latestWeight) return false;
      const ts = new Date(latestWeight.data.timestamp).getTime();
      // Only route if no previous timestamp OR 30s have passed
      return !lastRoutedWeightTs || ts - lastRoutedWeightTs > 10 * 1000;
    });

    if (deviceWithNewWeight) {
      const latestWeight = getLatestWeightForDevice(deviceWithNewWeight.mac)!;
      const ts = new Date(latestWeight.data.timestamp).getTime();

      // Update last routed timestamp
      setLastRoutedWeightTs(ts);

      // Only allow routing in device or tab screens (exclude logging)
      const inTabScreens = segments[0] === '(tabs)';
      const inDeviceScreens = segments[0] === '(device)';
      const inLogging = segments[1] === 'logging';
      if ((inTabScreens || inDeviceScreens) && !(inTabScreens && inLogging)) {
        router.push('/(body)/weight-taking');
      }
    }
  }, [connectedDevices, weightData]);

  return (
    <GluestackUIProvider mode={mode}>
      <Stack>
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="(auth)" options={{ headerShown: false }} />
        <Stack.Screen name="(body)" options={{ headerShown: false }} />
        <Stack.Screen name="(logging)" options={{ headerShown: false }} />
        <Stack.Screen name="(device)" options={{ headerShown: false }} />
        <Stack.Screen name="+not-found" options={{ headerShown: false }} />
      </Stack>
    </GluestackUIProvider>
  );
}
