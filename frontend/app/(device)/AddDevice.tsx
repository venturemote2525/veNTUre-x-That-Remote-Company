import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Button,
  RefreshControl,
  ScrollView,
} from 'react-native';
import Header from '@/components/Header';
import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import { AlertState } from '@/types/database-types';
import { CustomAlert } from '@/components/CustomAlert';
import { pairDevice } from '@/utils/device/api';
import { useAuth } from '@/context/AuthContext';

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
    getLatestWeightForDevice,
  } = useICDevice();
  const { profile } = useAuth();

  const [connecting, setConnecting] = useState<string | null>(null);
  const [disconnecting, setDisconnecting] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [scanDuration, setScanDuration] = useState(0);
  const [scanTimer, setScanTimer] = useState<number | null>(null);

  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });

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
        Alert.alert(
          'Scan Complete',
          'Scan stopped automatically after 30 seconds',
        );
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
      Alert.alert(
        'Initialization Failed',
        'Failed to initialize SDK. Please try again.',
      );
      console.error('SDK initialization error:', error);
    }
  };

  const handleConnectDevice = async (device: Device) => {
    setAlert({
      visible: true,
      title: 'Add device?',
      message: `Do you want to add ${device.name}?`,
      confirmText: 'Yes',
      onConfirm: () => handleConfirmConnectDevice(device),
      cancelText: 'No',
      onCancel: () => setAlert({ ...alert, visible: false }),
    });
  };

  const handleConfirmConnectDevice = async (device: Device) => {
    if (!profile) return;
    if (isDeviceConnected(device.mac)) {
      Alert.alert(
        'Already Connected',
        `Device ${device.mac} is already connected`,
      );
      return;
    }

    try {
      setConnecting(device.mac);
      // Add to database
      await pairDevice(profile.user_id, device.name ?? 'MY_SCALE', device.mac);
      // Add connection
      await connectDevice(device.mac);
      Alert.alert('Success', `Connected to ${device.mac}`);
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      Alert.alert(
        'Connection Failed',
        `Failed to connect to ${device.mac}\n\nError: ${errorMessage}`,
      );
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
      Alert.alert(
        'Disconnection Failed',
        `Failed to disconnect from ${device.mac}\n\nError: ${errorMessage}`,
      );
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
      Alert.alert(
        'Scan Failed',
        `Failed to start scanning\n\nError: ${errorMessage}`,
      );
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
        }>
        <View className="p-4">
          {/* SDK Status */}
          <View className="mb-4 rounded-lg bg-gray-50 p-4">
            <Text className="text-lg mb-2 font-semibold">SDK Status</Text>
            <View className="mb-2 flex-row items-center justify-between">
              <Text>Initialized:</Text>
              <Text
                className={
                  isSDKInitialized ? 'text-green-600' : 'text-red-600'
                }>
                {isSDKInitialized ? '✅ Ready' : '❌ Not Ready'}
              </Text>
            </View>
            <View className="mb-2 flex-row items-center justify-between">
              <Text>Scanning:</Text>
              <Text className={isScanning ? 'text-blue-600' : 'text-gray-600'}>
                {isScanning ? `🔍 Active (${scanDuration}s)` : '⏸️ Stopped'}
              </Text>
            </View>

            {!isSDKInitialized && (
              <Button title="Initialize SDK" onPress={handleInitializeSDK} />
            )}
          </View>

          {/* Weight Data Display */}
          {connectedDevices.length > 0 && (
            <View className="mb-4 rounded-lg bg-purple-50 p-4">
              <Text className="text-lg mb-2 font-semibold">
                Weight Measurements
              </Text>
              {connectedDevices.map(device => {
                const latestWeight = getLatestWeightForDevice(device.mac);
                const deviceWeights = weightData
                  .filter(m => m.device.mac === device.mac)
                  .slice(-3); // Show last 3 measurements

                return (
                  <View
                    key={device.mac}
                    className="mb-3 rounded-lg border bg-white p-3">
                    <Text className="text-sm mb-2 font-semibold">
                      Device: {device.mac}
                    </Text>

                    {latestWeight ? (
                      <View>
                        <Text className="text-2xl mb-1 font-bold text-purple-600">
                          {formatWeight(latestWeight.data.weight)} kg
                        </Text>
                        <Text className="text-xs mb-2 text-gray-500">
                          Last measurement:{' '}
                          {formatTime(latestWeight.data.timestamp)}
                          {latestWeight.data.isStabilized && ' (Stabilized)'}
                        </Text>

                        {deviceWeights.length > 1 && (
                          <View>
                            <Text className="text-sm mb-1 font-medium">
                              Recent measurements:
                            </Text>
                            {deviceWeights.reverse().map((measurement, idx) => (
                              <Text key={idx} className="text-xs text-gray-600">
                                {formatWeight(measurement.data.weight)}kg at{' '}
                                {formatTime(measurement.data.timestamp)}
                              </Text>
                            ))}
                          </View>
                        )}
                      </View>
                    ) : (
                      <Text className="italic text-gray-500">
                        No weight data yet - step on the scale to start
                        measuring
                      </Text>
                    )}
                  </View>
                );
              })}
            </View>
          )}

          {/* Scan Controls */}
          <View className="mb-4 rounded-lg bg-blue-50 p-4">
            <Text className="text-lg mb-2 font-semibold">Scan Control</Text>
            <View className="mb-2 flex-row gap-2">
              <View className="flex-1">
                <Button
                  title={
                    isScanning
                      ? `Stop Scan (${30 - scanDuration}s)`
                      : 'Start Scan'
                  }
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
              <Text className="text-sm text-center text-gray-600">
                Scan will auto-stop in {30 - scanDuration} seconds
              </Text>
            )}
          </View>

          {/* Scanned Devices */}
          <View className="mb-4">
            <Text className="text-lg mb-2 font-semibold">
              Available Devices ({scannedDevices.length})
            </Text>
            {scannedDevices.length === 0 ? (
              <View className="rounded-lg bg-gray-50 p-4">
                <Text className="text-center italic text-gray-500">
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
                    className={`mb-2 rounded-lg border p-4 ${
                      isConnected
                        ? 'border-green-200 bg-green-50'
                        : 'border-gray-200 bg-gray-50'
                    }`}>
                    <View className="mb-2 flex-row items-center justify-between">
                      <Text className="text-sm font-bold">
                        MAC: {device.mac}
                      </Text>
                      {device.rssi && (
                        <View className="flex-row items-center">
                          <Text className="text-xs mr-1">
                            {getRSSIIcon(device.rssi)}
                          </Text>
                          <Text className="text-xs">
                            {getRSSIText(device.rssi)} ({device.rssi}dBm)
                          </Text>
                        </View>
                      )}
                    </View>

                    <Text className="text-sm mb-3 text-gray-600">
                      Name: {device.name || 'Unknown Device'}
                    </Text>

                    <View className="flex-row items-center justify-between">
                      {isConnected ? (
                        <View className="flex-1">
                          <Text className="text-center font-medium text-green-600">
                            ✅ Connected
                          </Text>
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
                              <ActivityIndicator
                                size="small"
                                className="ml-2"
                              />
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
            <Text className="text-lg mb-2 font-semibold">
              Connected Devices ({connectedDevices.length})
            </Text>
            {connectedDevices.length === 0 ? (
              <View className="rounded-lg bg-gray-50 p-4">
                <Text className="text-center italic text-gray-500">
                  No devices connected
                </Text>
              </View>
            ) : (
              connectedDevices.map((device, index) => {
                const isDisconnectingThis = disconnecting === device.mac;

                return (
                  <View
                    key={device.mac || index}
                    className="mb-2 rounded-lg border border-green-200 bg-green-50 p-4">
                    <View className="mb-2 flex-row items-center justify-between">
                      <Text className="font-bold">MAC: {device.mac}</Text>
                      <Text className="font-medium text-green-600">
                        🟢 Active
                      </Text>
                    </View>

                    <Text className="text-sm mb-3 text-gray-600">
                      Name: {device.name || 'Unknown Device'}
                    </Text>

                    <Text className="mb-3 text-center font-medium text-green-600">
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
