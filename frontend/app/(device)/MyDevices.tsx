import Header from '@/components/Header';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { Pressable, ScrollView } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import { useRouter } from 'expo-router';
import { useICDevice } from '@/context/ICDeviceContext';
import { retrieveDevices } from '@/utils/device/api';
import { useFocusEffect } from '@react-navigation/native';
import { useCallback, useState } from 'react';
import { DBDevice } from '@/types/database-types';
import LoadingScreen from '@/components/LoadingScreen';
import BluetoothStatus from '@/components/devices/BluetoothStatus';
import LocationStatus from '@/components/devices/LocationStatus';

export default function MyDevices() {
  const {
    connectedDevices,
    weightData,
    isSDKInitialized,
    deviceBatteryLevels,
  } = useICDevice();

  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [pairedDevices, setPairedDevices] = useState<DBDevice[]>([]);

  useFocusEffect(
    useCallback(() => {
      let isActive = true;
      (async () => {
        try {
          setLoading(true);
          const data = await retrieveDevices();
          if (isActive) {
            setPairedDevices(data);
          }
        } catch (error) {
          console.error('Retrieve device error: ', error);
        } finally {
          if (isActive) setLoading(false);
        }
      })();
      return () => {
        isActive = false;
      };
    }, []),
  );

  if (loading) {
    return (
      <ThemedSafeAreaView>
        <LoadingScreen text="Loading devices" />
      </ThemedSafeAreaView>
    );
  }

  return (
    <ThemedSafeAreaView>
      <Header title="My Devices" />
      <View className="flex-1 gap-4 px-4">
        <BluetoothStatus />
        <LocationStatus />
        {/* SDK Status */}
        {!isSDKInitialized && (
          <View className="mb-4 rounded-lg bg-yellow-50 p-3">
            <Text className="text-yellow-800">
              ⚠️ SDK not initialized. Please check device permissions.
            </Text>
          </View>
        )}

        <ScrollView
          contentContainerStyle={{ flexGrow: 1, gap: 16 }}
          showsVerticalScrollIndicator={false}>
          {/* Paired Devices Section */}
          <View className="flex-1">
            <Text className="text-lg mb-3 font-bodyBold text-secondary-500">
              Paired Devices ({pairedDevices.length})
            </Text>

            {pairedDevices.length === 0 ? (
              <View className="flex-1 justify-center rounded-2xl bg-background-0 px-6 py-8">
                <Text className="text-center font-bodyBold text-primary-300">
                  No connected devices
                </Text>
                <Text className="text-sm mt-1 text-center text-gray-400">
                  Tap "Add Device" to connect your weight scale
                </Text>
              </View>
            ) : (
              pairedDevices.map((device, index) => {
                const isConnected = connectedDevices.some(
                  c => c.mac === device.mac,
                );
                return (
                  <Pressable
                    key={device.mac || index}
                    onPress={() =>
                      router.push({
                        pathname: '/(device)/DeviceInfo',
                        params: { deviceId: device.id },
                      })
                    }
                    className="mb-3 rounded-2xl bg-background-0 px-6 py-4">
                    <View className="flex-row items-center justify-between gap-4">
                      <View className="flex-1">
                        <View className="mb-1 flex-row items-center justify-between">
                          <Text className="font-bodyBold text-body2 text-secondary-500">
                            {device.name || 'Weight Scale'}
                          </Text>

                          <View className="rounded-full bg-secondary-500 px-2 py-1">
                            <Text className="font-bodySemiBold text-[10px] text-background-0">
                              {isConnected ? 'Connected' : 'Disconnected'}
                            </Text>
                          </View>
                        </View>
                        <View className="flex-row justify-between">
                          <Text className="text-sm font-bodySemiBold text-primary-300">
                            {device.mac}
                          </Text>
                          {isConnected && (
                            <View
                              className={`justify-center rounded-full px-2 ${deviceBatteryLevels[device.mac] > 75 ? 'bg-green-500' : deviceBatteryLevels[device.mac] > 30 ? 'bg-orange-500' : 'bg-red-500'}`}>
                              <Text className="font-bodySemiBold text-[10px] text-white">
                                {deviceBatteryLevels[device.mac]} %
                              </Text>
                            </View>
                          )}
                        </View>
                      </View>

                      <AntDesign name="caretright" size={20} color="#6B7280" />
                    </View>
                  </Pressable>
                );
              })
            )}
          </View>

          {/* Recent Activity Section */}
          {weightData.length > 0 && (
            <View>
              <Text className="text-lg mb-3 font-bodyBold text-secondary-500">
                Recent Activity
              </Text>

              {weightData
                .slice(-3) // Show last 3 measurements
                .reverse()
                .map((measurement, index) => (
                  <View
                    key={index}
                    className="mb-2 rounded-lg bg-gray-50 px-4 py-3">
                    <View className="flex-row items-center justify-between">
                      <View>
                        <Text className="font-bodyBold text-secondary-500">
                          {measurement.data.weight.toFixed(2)}kg
                        </Text>
                        <Text className="text-xs text-gray-400">
                          {measurement.device?.mac}
                        </Text>
                      </View>
                      <Text className="text-xs text-gray-400">
                        {new Date(measurement.timestamp).toLocaleString()}
                      </Text>
                    </View>
                  </View>
                ))}
            </View>
          )}

          {/* Device Statistics */}
          {connectedDevices.length > 0 && (
            <View className="rounded-2xl bg-blue-50 px-6 py-4">
              <Text className="mb-2 font-bodyBold text-secondary-500">
                📊 Quick Stats
              </Text>
              <View className="flex-row justify-between">
                <View>
                  <Text className="text-sm text-gray-600">Total Devices</Text>
                  <Text className="text-lg font-bodyBold">
                    {connectedDevices.length}
                  </Text>
                </View>
                <View>
                  <Text className="text-sm text-gray-600">Measurements</Text>
                  <Text className="text-lg font-bodyBold">
                    {weightData.length}
                  </Text>
                </View>
                <View>
                  <Text className="text-sm text-gray-600">Today</Text>
                  <Text className="text-lg font-bodyBold">
                    {
                      weightData.filter(
                        w =>
                          new Date(w.timestamp).toDateString() ===
                          new Date().toDateString(),
                      ).length
                    }
                  </Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Add Device Button - Fixed at bottom */}
        <View className="mb-4">
          <Pressable
            onPress={() => router.push('/(device)/AddDevice')}
            className="button-rounded">
            <View className="flex-row items-center justify-center">
              <AntDesign
                name="plus"
                size={20}
                color="white"
                style={{ marginRight: 8 }}
              />
              <Text className="font-bodySemiBold text-background-0">
                Add Device
              </Text>
            </View>
          </Pressable>
        </View>
      </View>
    </ThemedSafeAreaView>
  );
}
