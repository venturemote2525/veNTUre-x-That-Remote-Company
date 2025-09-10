import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import Header from '@/components/Header';
import { useAuth } from '@/context/AuthContext';
import { useICDevice } from '@/context/ICDeviceContext';
import { AlertState, DBDevice } from '@/types/database-types';
import { Device } from '@/types/icdevice-types';
import { retrieveDeviceInfo, unpairDevice } from '@/utils/device/api';
import LoadingScreen from '@/components/LoadingScreen';
import { Alert, Pressable } from 'react-native';
import { CustomAlert } from '@/components/CustomAlert';

export default function DeviceInfo() {
  const { deviceId } = useLocalSearchParams();
  const deviceIdStr = Array.isArray(deviceId) ? deviceId[0] : deviceId;
  const { profile } = useAuth();
  const {
    disconnectDevice,
    refreshDevices,
    isDeviceConnected,
    setPairedDevices,
  } = useICDevice();
  const router = useRouter();
  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });
  const [device, setDevice] = useState<DBDevice | null>(null);

  useEffect(() => {
    const fetchDeviceInfo = async () => {
      try {
        const data = await retrieveDeviceInfo(deviceIdStr);
        setDevice(data);
      } catch (error) {
        console.log(error);
      }
    };
    fetchDeviceInfo();
  }, [deviceId]);

  const handleRemoveDevice = (device: Device) => {
    setAlert({
      visible: true,
      title: 'Remove device?',
      message: `Do you want to add ${device.name}?`,
      confirmText: 'Yes',
      onConfirm: () => handleConfirmRemove(device),
      cancelText: 'No',
      onCancel: () => setAlert({ ...alert, visible: false }),
    });
  };

  const handleConfirmRemove = async (device: Device) => {
    if (!profile) return;
    try {
      // 1. Remove device from database
      await unpairDevice(profile.user_id, device.mac);
      // 2. If the device is connected, disconnect it
      if (await isDeviceConnected(device.mac)) {
        await disconnectDevice(device.mac);
      } else {
        setPairedDevices(prev => prev.filter(d => d.mac !== device.mac));
      }
      // 3. Refresh devices
      await refreshDevices();
      setAlert({ ...alert, visible: false });
      router.back();
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      setAlert({
        visible: true,
        title: 'Disconnection Failed',
        message: `Failed to disconnect from ${device.mac}\n\nError: ${errorMessage}`,
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
      });
      console.error('Disconnection error:', error);
    }
  };

  return (
    <ThemedSafeAreaView>
      {device ? (
        <View>
          <Header title={device.name} />
          <View className="px-4">
            <Text>MAC: {device.mac}</Text>
            <Pressable
              className="button-rounded"
              onPress={() => handleRemoveDevice(device)}>
              <Text>Remove Device</Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <LoadingScreen text="Loading device" />
      )}

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
