import Header from '@/components/Header';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { NativeEventEmitter, NativeModules, Button, Alert, FlatList } from 'react-native'
import { useState, useEffect } from 'react'

const { ICDeviceModule } = NativeModules;
const deviceEmitter = new NativeEventEmitter(ICDeviceModule);

export default function AddDevice() {
    const [devices, setDevices] = useState([]);

    useEffect(() => {
        console.log('Start scan')
        const timeout = setTimeout(() => {
            ICDeviceModule.initializeSDK();
        }, 500); // delay to ensure activity exists
        return () => clearTimeout(timeout);
    }, []);

    const handleStartScan = () => {
        ICDeviceModule.startScan();
        Alert.alert("Scan started");
      };

      const handleStopScan = () => {
        ICDeviceModule.stopScan();
        Alert.alert("Scan stopped");
      };

      const handleAddDevice = () => {
        ICDeviceModule.addDevice();
        Alert.alert("Add device called");
      };
  return (
    <ThemedSafeAreaView>
      <Header title="Add Device" />
      <View className="flex-1 items-center justify-center">
        <Text>Instructions on how to add device here</Text>
        <Button title="Start Scan" onPress={handleStartScan} />
          <Button title="Stop Scan" onPress={handleStopScan} />
      </View>
    </ThemedSafeAreaView>
  );
}
