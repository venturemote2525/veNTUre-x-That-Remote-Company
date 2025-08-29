import Header from '@/components/Header';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import { useEffect, useMemo, useState } from 'react';
import { Button, FlatList, Pressable } from 'react-native';
import { Device } from '@/types/icdevice-types';
import { pairDevice } from '@/utils/device/api';
import { useAuth } from '@/context/AuthContext';
import { CustomAlert } from '@/components/CustomAlert';
import { useRouter } from 'expo-router';
import { AlertState } from '@/types/database-types';
import BluetoothStatus from "@/components/devices/BluetoothStatus";

export default function AddDevice() {
  const router = useRouter();
  const { profile } = useAuth();
  const {
    scannedDevices,
    connectedDevices,
    startScan,
    stopScan,
    addDevice,
    bluetoothEnabled,
  } = useICDevice();
  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });
    const [availableDevices, setAvailableDevices] = useState<Device[]>([]);

  useEffect(() => {
    startScan();
    return () => stopScan();
  }, []);

    useEffect(() => {
        setAvailableDevices(
            scannedDevices.filter(d => !connectedDevices.some(c => c.mac === d.mac))
        );
    }, [scannedDevices, connectedDevices]);

  const handleAddDevice = (device: Device) => {
    setAlert({
      visible: true,
      title: 'Add device?',
      message: `Do you want to add ${device.name}?`,
      confirmText: 'Yes',
      onConfirm: () => handleConfirmAdd(device),
      cancelText: 'No',
      onCancel: () => setAlert({ ...alert, visible: false }),
    });
  };

  const handleConfirmAdd = async (device: Device) => {
    if (!profile) return;
    try {
      await pairDevice(profile.user_id, device.name ?? 'MY_SCALE', device.mac);
      addDevice(device);
      setAlert({ ...alert, visible: false });
      router.back();
    } catch (error) {
      console.error('Add device error', error);
      setAlert({
        visible: true,
        title: 'Error',
        message: 'Error adding device',
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
      });
    }
  };

  return (
    <ThemedSafeAreaView>
      <Header title="Add Device" />
      <View className="flex-1 px-4 gap-4">

          <BluetoothStatus />
        {/* Instructions */}
          <View className="items-center">
        <Text>Instructions on how to add device here</Text>
          </View>
        {/* List of scanned devices */}
        <FlatList
          className="flex-1 w-full"
          data={availableDevices}
          keyExtractor={item => item.mac}
          renderItem={({ item }) => (
            <Pressable
              className="rounded-2xl bg-background-0 px-6 py-4 w-full"
              onPress={() => handleAddDevice(item)}>
              <Text className="font-bodyBold text-body2 text-secondary-500">
                {item.name}
              </Text>
              <Text className="font-bodySemiBold text-primary-300">
                {item.mac}
              </Text>
            </Pressable>
          )}
        />
      </View>

      <CustomAlert
        visible={alert.visible}
        title={alert.title}
        message={alert.message}
        confirmText={alert.confirmText}
        onConfirm={alert.onConfirm ?? (() => {})}
        cancelText={alert.cancelText}
        onCancel={alert.onCancel}
      />
    </ThemedSafeAreaView>
  );
}
