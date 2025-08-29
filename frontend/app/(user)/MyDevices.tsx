import Header from '@/components/Header';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { Pressable, ScrollView } from 'react-native';
import AntDesign from '@expo/vector-icons/AntDesign';
import { useFocusEffect, useRouter } from 'expo-router';
import { useICDevice } from '@/context/ICDeviceContext';
import { useState, useCallback } from 'react';
import { DBDevice } from '@/types/database-types';
import { retrieveDevices, } from '@/utils/device/api';
import LoadingScreen from '@/components/LoadingScreen';
import BluetoothStatus from "@/components/devices/BluetoothStatus";

export default function MyDevices() {
  const { connectedDevices, removeDevice } = useICDevice();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [pairedDevices, setPairedDevices] = useState<DBDevice[]>([]);

  useFocusEffect(
    useCallback(() => {
      const fetchPairedDevices = async () => {
        try {
          setLoading(true);
          const data = await retrieveDevices();
          setPairedDevices(data);
        } catch (error) {
          console.error('Retrieve device error: ', error);
        } finally {
          setLoading(false);
        }
      };
      fetchPairedDevices();
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
      <View className="flex-1 px-4 gap-4">
          <BluetoothStatus />
        <ScrollView contentContainerStyle={{ gap: 16 }}>
          {pairedDevices.map(device => {
            const isConnected = connectedDevices.some(
              c => c.mac === device.mac,
            );
            return (
              <Pressable
                key={device.id}
                onPress={() =>
                  router.push({
                    pathname: '/(user)/DeviceInfo',
                    params: { deviceId: device.id },
                  })
                }
                className="rounded-2xl bg-background-0 px-6 py-4">
                <View className="flex-row items-center justify-between gap-4">
                  <View className="flex-1 flex-col">
                    <View className="flex-row items-center justify-between">
                      <Text className="font-bodyBold text-body2 text-secondary-500">
                        {device.name}
                      </Text>
                      <View className="rounded-full bg-secondary-500 px-2 py-1">
                        <Text className="font-bodySemiBold text-[10px] text-background-0">
                          {isConnected ? 'Connected' : 'Disconnected'}
                        </Text>
                      </View>
                    </View>
                    <Text className="font-bodySemiBold text-primary-300">
                      {device.mac}
                    </Text>
                  </View>
                  <AntDesign name="caretright" size={20} />
                </View>
              </Pressable>
            );
          })}
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
