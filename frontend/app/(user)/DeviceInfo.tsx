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
import { Pressable } from 'react-native';
import { CustomAlert } from '@/components/CustomAlert';

export default function DeviceInfo() {
  const { deviceId } = useLocalSearchParams()
  const deviceIdStr = Array.isArray(deviceId) ? deviceId[0] : deviceId;
  const { profile } = useAuth();
  const { removeDevice } = useICDevice();
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
    }
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
      await unpairDevice(profile.user_id, device.mac);
      removeDevice(device.mac);
      setAlert({ ...alert, visible: false });
      router.back();
    } catch (error) {
      console.log('Remove device error: ', error);
      setAlert({
        visible: true,
        title: 'Error',
        message: 'Error removing device',
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
      });
    }
  };

  return (
    <ThemedSafeAreaView>
      {device ? (
        <View>
          <Header title={device.name} />
          <View className='px-4'>
            <Text>MAC: {device.mac}</Text>
            <Pressable className='button-rounded' onPress={() => handleRemoveDevice(device)}>
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
  )
}