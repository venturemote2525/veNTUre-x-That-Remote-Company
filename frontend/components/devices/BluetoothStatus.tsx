import { View, Text } from '@/components/Themed'
import { useICDevice } from "@/context/ICDeviceContext"

export default function BluetoothStatus() {
    const { bluetoothEnabled } = useICDevice();

    if (bluetoothEnabled) return null;

    return (
        <View className="rounded-full bg-secondary-500 px-4 py-1 items-center">
            <Text className="font-bodySemiBold text-background-500">
                Bluetooth not enabled
            </Text>
        </View>
    );
}
