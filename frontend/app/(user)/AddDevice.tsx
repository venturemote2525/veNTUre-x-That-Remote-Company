import React, { useState, useEffect } from 'react';
import { Button, Alert, ActivityIndicator, RefreshControl, ScrollView } from 'react-native';
import Header from '@/components/Header';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';

interface Device {
  mac: string;
  name?: string;
  rssi?: number;
  isConnected?: boolean;
}

export default function AddDevice() {
  const {
    scannedDevices,
    connectedDevices,
    weightData,
    isScanning,
    isSDKInitialized,
    connectDevice,
    disconnectDevice,
    startScan,
    stopScan,
    initializeSDK,
    clearScannedDevices,
    getLatestWeightForDevice
  } = useICDevice();

  const [connecting, setConnecting] = useState<string | null>(null);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [scanDuration, setScanDuration] = useState(0);
  const [scanTimer, setScanTimer] = useState<NodeJS.Timeout | null>(null);

  // Auto-stop scan after 30 seconds
  useEffect(() => {
    if (isScanning) {
      const timer = setInterval(() => {
        setScanDuration(prev => prev + 1);
      }, 1000);
      setScanTimer(timer);

      // Auto stop after 30 seconds
      const autoStopTimer = setTimeout(() => {
        handleStopScan();
        Alert.alert('Scan Complete', 'Scan stopped automatically after 30 seconds');
      }, 30000);

      return () => {
        clearInterval(timer);
        clearTimeout(autoStopTimer);
      };
    } else {
      setScanDuration(0);
      if (scanTimer) {
        clearInterval(scanTimer);
        setScanTimer(null);
      }
    }
  }, [isScanning]);

  const handleInitializeSDK = async () => {
    try {
      await initializeSDK();
      Alert.alert('Success', 'SDK initialized successfully');
    } catch (error) {
      Alert.alert('Initialization Failed', 'Failed to initialize SDK. Please try again.');
      console.error('SDK initialization error:', error);
    }
  };

  const handleConnectDevice = async (device: Device) => {
    if (isDeviceConnected(device.mac)) {
      Alert.alert('Already Connected', `Device ${device.mac} is already connected`);
      return;
    }

    try {
      setConnecting(device.mac);
      await connectDevice(device.mac);
      Alert.alert('Success', `Connected to ${device.mac}`);
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      Alert.alert('Connection Failed', `Failed to connect to ${device.mac}\n\nError: ${errorMessage}`);
      console.error('Connection error:', error);
    } finally {
      setConnecting(null);
    }
  };

  const handleDisconnectDevice = async (device: Device) => {
    try {
      setDisconnecting(device.mac);
      await disconnectDevice(device.mac);
      Alert.alert('Disconnected', `Disconnected from ${device.mac}`);
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      Alert.alert('Disconnection Failed', `Failed to disconnect from ${device.mac}\n\nError: ${errorMessage}`);
      console.error('Disconnection error:', error);
    } finally {
      setDisconnecting(null);
    }
  };

  const handleStartScan = async () => {
    if (!isSDKInitialized) {
      Alert.alert('SDK Not Ready', 'Please initialize the SDK first');
      return;
    }

    try {
      await startScan();
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      Alert.alert('Scan Failed', `Failed to start scanning\n\nError: ${errorMessage}`);
      console.error('Scan start error:', error);
    }
  };

  const handleStopScan = async () => {
    try {
      await stopScan();
    } catch (error: any) {
      console.error('Scan stop error:', error);
    }
  };

  const handleClearDevices = async () => {
    try {
      await clearScannedDevices();
      Alert.alert('Cleared', 'Scanned devices list cleared');
    } catch (error) {
      console.error('Clear devices error:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      if (isScanning) {
        await handleStopScan();
      }
      await handleClearDevices();
      await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay
    } finally {
      setRefreshing(false);
    }
  };

  const isDeviceConnected = (mac: string): boolean => {
    return connectedDevices.some(device => device.mac === mac);
  };

  const getRSSIIcon = (rssi?: number): string => {
    if (!rssi) return '📶';
    if (rssi > -50) return '📶';
    if (rssi > -70) return '📵';
    return '📴';
  };

  const getRSSIText = (rssi?: number): string => {
    if (!rssi) return 'Unknown';
    if (rssi > -50) return 'Excellent';
    if (rssi > -70) return 'Good';
    if (rssi > -80) return 'Fair';
    return 'Poor';
  };

  const formatWeight = (weight: number): string => {
    return weight.toFixed(2);
  };

  const formatTime = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <ThemedSafeAreaView>
      <Header title="Add Device" />
      <ScrollView
        className="flex-1"
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        <View className="p-4">
          {/* SDK Status */}
          <View className="mb-4 p-4 bg-gray-50 rounded-lg">
            <Text className="text-lg font-semibold mb-2">SDK Status</Text>
            <View className="flex-row items-center justify-between mb-2">
              <Text>Initialized:</Text>
              <Text className={isSDKInitialized ? 'text-green-600' : 'text-red-600'}>
                {isSDKInitialized ? '✅ Ready' : '❌ Not Ready'}
              </Text>
            </View>
            <View className="flex-row items-center justify-between mb-2">
              <Text>Scanning:</Text>
              <Text className={isScanning ? 'text-blue-600' : 'text-gray-600'}>
                {isScanning ? `🔍 Active (${scanDuration}s)` : '⏸️ Stopped'}
              </Text>
            </View>

            {!isSDKInitialized && (
              <Button
                title="Initialize SDK"
                onPress={handleInitializeSDK}
              />
            )}
          </View>

          {/* Weight Data Display */}
          {connectedDevices.length > 0 && (
            <View className="mb-4 p-4 bg-purple-50 rounded-lg">
              <Text className="text-lg font-semibold mb-2">Weight Measurements</Text>
              {connectedDevices.map((device) => {
                const latestWeight = getLatestWeightForDevice(device.mac);
                const deviceWeights = weightData
                  .filter(m => m.device.mac === device.mac)
                  .slice(-3); // Show last 3 measurements

                return (
                  <View key={device.mac} className="mb-3 p-3 bg-white rounded-lg border">
                    <Text className="font-semibold text-sm mb-2">Device: {device.mac}</Text>

                    {latestWeight ? (
                      <View>
                        <Text className="text-2xl font-bold text-purple-600 mb-1">
                          {formatWeight(latestWeight.data.weight)} kg
                        </Text>
                        <Text className="text-xs text-gray-500 mb-2">
                          Last measurement: {formatTime(latestWeight.data.timestamp)}
                          {latestWeight.data.isStabilized && " (Stabilized)"}
                        </Text>

                        {deviceWeights.length > 1 && (
                          <View>
                            <Text className="text-sm font-medium mb-1">Recent measurements:</Text>
                            {deviceWeights.reverse().map((measurement, idx) => (
                              <Text key={idx} className="text-xs text-gray-600">
                                {formatWeight(measurement.data.weight)}kg at {formatTime(measurement.data.timestamp)}
                              </Text>
                            ))}
                          </View>
                        )}
                      </View>
                    ) : (
                      <Text className="text-gray-500 italic">
                        No weight data yet - step on the scale to start measuring
                      </Text>
                    )}
                  </View>
                );
              })}
            </View>
          )}

          {/* Scan Controls */}
          <View className="mb-4 p-4 bg-blue-50 rounded-lg">
            <Text className="text-lg font-semibold mb-2">Scan Control</Text>
            <View className="flex-row gap-2 mb-2">
              <View className="flex-1">
                <Button
                  title={isScanning ? `Stop Scan (${30 - scanDuration}s)` : "Start Scan"}
                  onPress={isScanning ? handleStopScan : handleStartScan}
                  disabled={!isSDKInitialized}
                />
              </View>
              <View className="flex-1">
                <Button
                  title="Clear List"
                  onPress={handleClearDevices}
                  disabled={scannedDevices.length === 0}
                />
              </View>
            </View>
            {isScanning && (
              <Text className="text-sm text-gray-600 text-center">
                Scan will auto-stop in {30 - scanDuration} seconds
              </Text>
            )}
          </View>

          {/* Scanned Devices */}
          <View className="mb-4">
            <Text className="text-lg font-semibold mb-2">
              Available Devices ({scannedDevices.length})
            </Text>
            {scannedDevices.length === 0 ? (
              <View className="p-4 bg-gray-50 rounded-lg">
                <Text className="text-gray-500 italic text-center">
                  {isScanning
                    ? 'Scanning for devices...'
                    : 'No devices found. Start scanning to find devices.'}
                </Text>
                {isScanning && (
                  <ActivityIndicator size="large" className="mt-2" />
                )}
              </View>
            ) : (
              scannedDevices.map((device, index) => {
                const isConnected = isDeviceConnected(device.mac);
                const isConnectingThis = connecting === device.mac;

                return (
                  <View
                    key={device.mac || index}
                    className={`p-4 mb-2 rounded-lg border ${
                      isConnected ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                    }`}
                  >
                    <View className="flex-row items-center justify-between mb-2">
                      <Text className="font-bold text-sm">MAC: {device.mac}</Text>
                      {device.rssi && (
                        <View className="flex-row items-center">
                          <Text className="text-xs mr-1">{getRSSIIcon(device.rssi)}</Text>
                          <Text className="text-xs">{getRSSIText(device.rssi)} ({device.rssi}dBm)</Text>
                        </View>
                      )}
                    </View>

                    <Text className="text-sm text-gray-600 mb-3">
                      Name: {device.name || 'Unknown Device'}
                    </Text>

                    <View className="flex-row items-center justify-between">
                      {isConnected ? (
                        <View className="flex-1">
                          <Text className="text-green-600 font-medium text-center">✅ Connected</Text>
                        </View>
                      ) : (
                        <View className="flex-1">
                          <View className="flex-row items-center justify-center">
                            <Button
                              title="Connect"
                              onPress={() => handleConnectDevice(device)}
                              disabled={isConnectingThis}
                            />
                            {isConnectingThis && (
                              <ActivityIndicator size="small" className="ml-2" />
                            )}
                          </View>
                        </View>
                      )}
                    </View>
                  </View>
                );
              })
            )}
          </View>

          {/* Connected Devices */}
          <View className="mb-4">
            <Text className="text-lg font-semibold mb-2">
              Connected Devices ({connectedDevices.length})
            </Text>
            {connectedDevices.length === 0 ? (
              <View className="p-4 bg-gray-50 rounded-lg">
                <Text className="text-gray-500 italic text-center">No devices connected</Text>
              </View>
            ) : (
              connectedDevices.map((device, index) => {
                const isDisconnectingThis = disconnecting === device.mac;

                return (
                  <View key={device.mac || index} className="p-4 mb-2 bg-green-50 border border-green-200 rounded-lg">
                    <View className="flex-row items-center justify-between mb-2">
                      <Text className="font-bold">MAC: {device.mac}</Text>
                      <Text className="text-green-600 font-medium">🟢 Active</Text>
                    </View>

                    <Text className="text-sm text-gray-600 mb-3">
                      Name: {device.name || 'Unknown Device'}
                    </Text>

                    <Text className="text-green-600 font-medium mb-3 text-center">
                      📊 Ready to receive weight data
                    </Text>

                    <View className="flex-row items-center justify-center">
                      <Button
                        title="Disconnect"
                        onPress={() => handleDisconnectDevice(device)}
                        disabled={isDisconnectingThis}
                        color="#dc3545"
                      />
                      {isDisconnectingThis && (
                        <ActivityIndicator size="small" className="ml-2" />
                      )}
                    </View>
                  </View>
                );
              })
            )}
          </View>
        </View>
      </ScrollView>
    </ThemedSafeAreaView>
  );
}