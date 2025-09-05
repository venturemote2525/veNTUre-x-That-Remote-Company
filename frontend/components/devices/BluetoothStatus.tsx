import { Text, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';

export default function BluetoothStatus() {
  const { bleEnabled } = useICDevice();

  if (bleEnabled) return null;

  return (
    <View className="items-center rounded-full bg-secondary-500 px-4 py-1">
      <Text className="font-bodySemiBold text-background-500">
        Bluetooth not enabled
      </Text>
    </View>
  );
}
