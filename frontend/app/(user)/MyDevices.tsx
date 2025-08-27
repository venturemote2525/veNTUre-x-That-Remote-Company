import Header from '@/components/Header';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { Pressable, ScrollView } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import { useRouter } from 'expo-router';
import { useICDevice } from '@/context/ICDeviceContext';

export default function MyDevices() {
  const { connectedDevices } = useICDevice();
  const router = useRouter();
  const devices = [
    { id: 1, name: 'device 1', mac: 'mac 1' },
    { id: 2, name: 'device 2', mac: 'mac 2' },
  ];
  return (
    <ThemedSafeAreaView>
      <Header title="My Devices" />
      <View className="flex-1 px-4">
        <ScrollView contentContainerStyle={{ gap: 16 }}>
          {devices.map(device => (
            <Pressable
              key={device.id}
              onPress={() => console.log('Pressed: device ', device.name)}
              className="rounded-2xl bg-background-0 px-6 py-4">
              <View className="flex-row items-center justify-between">
                <View>
                  <Text className="font-bodyBold text-body2 text-secondary-500">
                    {device.name}
                  </Text>
                  <Text className="font-bodySemiBold text-primary-300">
                    {device.mac}
                  </Text>
                </View>
                <Text className="text-secondary-500">
                  <AntDesign name="caretright" size={20} />
                </Text>
              </View>
            </Pressable>
          ))}
        </ScrollView>
        <ScrollView contentContainerStyle={{ gap: 16 }}>
          {connectedDevices.length === 0 ? (
            <Text className="text-primary-300">No connected devices</Text>
          ) : (
            connectedDevices.map((device, index) => (
              <Pressable
                key={index}
                onPress={() => console.log('Pressed: device ', device.mac)}
                className="rounded-2xl bg-background-0 px-6 py-4">
                <View className="flex-row items-center justify-between">
                  <View>
                    <Text className="font-bodyBold text-body2 text-secondary-500">
                      {device.name || 'Unknown Device'}
                    </Text>
                    <Text className="font-bodySemiBold text-primary-300">
                      {device.mac}
                    </Text>
                  </View>
                  <AntDesign name="caretright" size={20} color="#6B7280" />
                </View>
              </Pressable>
            ))
          )}
        </ScrollView>
        <Pressable
          onPress={() => router.push('/(user)/AddDevice')}
          className="button-rounded my-4">
          <Text className="font-bodySemiBold text-background-0">
            Add Device
          </Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
