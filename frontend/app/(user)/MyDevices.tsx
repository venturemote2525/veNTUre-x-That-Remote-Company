import Header from '@/components/Header';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { Pressable, ScrollView } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import { useRouter } from 'expo-router';
import { useICDevice } from '@/context/ICDeviceContext';

export default function MyDevices() {
  const {
    connectedDevices,
    weightData,
    getLatestWeightForDevice,
    isSDKInitialized
  } = useICDevice();

  const router = useRouter();

  const handleDevicePress = (device: any) => {
    console.log('Pressed device:', device.mac);
    // Navigate to device details page
    // router.push(`/(user)/DeviceDetails?mac=${device.mac}`);
  };

  const getDeviceStatus = (mac: string) => {
    const latestWeight = getLatestWeightForDevice(mac);
    if (latestWeight) {
      const timeAgo = Math.floor((Date.now() - latestWeight.timestamp) / 1000 / 60); // minutes ago
      return `${latestWeight.data.weight}kg • ${timeAgo}m ago`;
    }
    return 'No recent data';
  };

  const getStatusColor = (mac: string) => {
    const latestWeight = getLatestWeightForDevice(mac);
    if (latestWeight) {
      const hoursAgo = (Date.now() - latestWeight.timestamp) / 1000 / 60 / 60;
      return hoursAgo < 24 ? 'text-green-600' : 'text-yellow-600';
    }
    return 'text-gray-400';
  };

  return (
    <ThemedSafeAreaView>
      <Header title="My Devices" />
      <View className="flex-1 px-4">
        {/* SDK Status */}
        {!isSDKInitialized && (
          <View className="mb-4 p-3 bg-yellow-50 rounded-lg">
            <Text className="text-yellow-800">
              ⚠️ SDK not initialized. Please check device permissions.
            </Text>
          </View>
        )}

        <ScrollView
          contentContainerStyle={{ gap: 16, paddingBottom: 100 }}
          showsVerticalScrollIndicator={false}
        >
          {/* Connected Devices Section */}
          <View>
            <Text className="font-bodyBold text-lg text-secondary-500 mb-3">
              Connected Devices ({connectedDevices.length})
            </Text>

            {connectedDevices.length === 0 ? (
              <View className="rounded-2xl bg-background-0 px-6 py-8">
                <Text className="text-center text-primary-300">
                  No connected devices
                </Text>
                <Text className="text-center text-sm text-gray-400 mt-1">
                  Tap "Add Device" to connect your weight scale
                </Text>
              </View>
            ) : (
              connectedDevices.map((device, index) => (
                <Pressable
                  key={device.mac || index}
                  onPress={() => handleDevicePress(device)}
                  className="rounded-2xl bg-background-0 px-6 py-4 mb-3"
                >
                  <View className="flex-row items-center justify-between">
                    <View className="flex-1">
                      <View className="flex-row items-center mb-1">
                        <View className="w-2 h-2 bg-green-500 rounded-full mr-2" />
                        <Text className="font-bodyBold text-body2 text-secondary-500">
                          {device.name || 'Weight Scale'}
                        </Text>
                      </View>

                      <Text className="font-bodySemiBold text-primary-300 text-sm">
                        {device.mac}
                      </Text>

                      <Text className={`text-xs mt-1 ${getStatusColor(device.mac)}`}>
                        {getDeviceStatus(device.mac)}
                      </Text>
                    </View>

                    <AntDesign name="caretright" size={20} color="#6B7280" />
                  </View>
                </Pressable>
              ))
            )}
          </View>

          {/* Recent Activity Section */}
          {weightData.length > 0 && (
            <View>
              <Text className="font-bodyBold text-lg text-secondary-500 mb-3">
                Recent Activity
              </Text>

              {weightData
                .slice(-3) // Show last 3 measurements
                .reverse()
                .map((measurement, index) => (
                  <View
                    key={index}
                    className="rounded-lg bg-gray-50 px-4 py-3 mb-2"
                  >
                    <View className="flex-row justify-between items-center">
                      <View>
                        <Text className="font-bodyBold text-secondary-500">
                          {measurement.data.weight}kg
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
                ))
              }
            </View>
          )}

          {/* Device Statistics */}
          {connectedDevices.length > 0 && (
            <View className="rounded-2xl bg-blue-50 px-6 py-4">
              <Text className="font-bodyBold text-secondary-500 mb-2">
                📊 Quick Stats
              </Text>
              <View className="flex-row justify-between">
                <View>
                  <Text className="text-sm text-gray-600">Total Devices</Text>
                  <Text className="font-bodyBold text-lg">{connectedDevices.length}</Text>
                </View>
                <View>
                  <Text className="text-sm text-gray-600">Measurements</Text>
                  <Text className="font-bodyBold text-lg">{weightData.length}</Text>
                </View>
                <View>
                  <Text className="text-sm text-gray-600">Today</Text>
                  <Text className="font-bodyBold text-lg">
                    {weightData.filter(w =>
                      new Date(w.timestamp).toDateString() === new Date().toDateString()
                    ).length}
                  </Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>

        {/* Add Device Button - Fixed at bottom */}
        <View className="absolute bottom-4 left-4 right-4">
          <Pressable
            onPress={() => router.push('/(user)/AddDevice')}
            className="button-rounded"
          >
            <View className="flex-row items-center justify-center">
              <AntDesign name="plus" size={20} color="white" style={{ marginRight: 8 }} />
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
