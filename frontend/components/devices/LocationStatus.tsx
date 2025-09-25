// components/devices/LocationStatus.tsx
import { useEffect, useState } from 'react';
import { View, Pressable, Linking } from 'react-native';
import { Text } from '@/components/Themed';
import * as Location from 'expo-location';

export default function LocationStatus() {
  const [locationEnabled, setLocationEnabled] = useState<boolean | null>(null);

  const checkLocationPermissions = async () => {
    try {
      const servicesEnabled = await Location.hasServicesEnabledAsync();
      if (!servicesEnabled) {
        setLocationEnabled(false);
        return;
      }

      const { status } = await Location.getForegroundPermissionsAsync();
      setLocationEnabled(status === 'granted');
    } catch (error) {
      console.error('Error checking location permissions:', error);
      setLocationEnabled(false);
    }
  };

  const handleOpenSettings = () => {
    Linking.openSettings();
  };

  useEffect(() => {
    checkLocationPermissions();
  }, []);

  if (locationEnabled === null) {
    return (
      <View className="mb-4 rounded-lg bg-gray-100 p-3">
        <Text className="text-gray-600">Checking location status...</Text>
      </View>
    );
  }

  return (
    <View
      className={`mb-4 rounded-lg p-3 ${
        locationEnabled ? 'bg-green-50' : 'bg-red-50'
      }`}>
      <View className="flex-row justify-between items-center">
        <Text
          className={`font-bodySemiBold ${
            locationEnabled ? 'text-green-700' : 'text-red-700'
          }`}>
          {locationEnabled
            ? '✅ Location services enabled'
            : '⚠️ Location services disabled'}
        </Text>

        {!locationEnabled && (
          <Pressable
            onPress={handleOpenSettings}
            className="rounded-lg bg-red-600 px-3 py-1">
            <Text className="text-white text-sm font-bodySemiBold">
              Open Settings
            </Text>
          </Pressable>
        )}
      </View>
    </View>
  );
}
