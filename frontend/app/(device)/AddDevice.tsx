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
import LoadingScreen from '@/components/LoadingScreen';

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
    isScanning,
    isSDKInitialized,
    connectDevice,
    startScan,
    stopScan,
    clearScannedDevices,
    refreshDevices,
    bleEnabled,
  } = useICDevice();
  const { profile } = useAuth();

  const [connecting, setConnecting] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });

  // Start scan on mount and stop when unmount
  useEffect(() => {
    (async () => {
      if (isSDKInitialized) {
        try {
          await startScan();
        } catch (error) {
          console.error('Failed to start scan:', error);
        }
      }
    })();
    return () => {
      (async () => {
        try {
          await stopScan();
        } catch (error) {
          console.error('Failed to stop scan:', error);
        }
      })();
    };
  }, [isSDKInitialized, startScan, stopScan]);

  const handleConnectDevice = async (device: Device) => {
    setAlert({
      visible: true,
      title: 'Add device?',
      message: `Do you want to add ${device.mac}?`,
      confirmText: 'Yes',
      onConfirm: () => {
        handleConfirmConnectDevice(device);
        setAlert({ ...alert, visible: false });
      },
      cancelText: 'No',
      onCancel: () => setAlert({ ...alert, visible: false }),
    });
  };

  const handleConfirmConnectDevice = async (device: Device) => {
    if (!profile) return;
    if (isDeviceConnected(device.mac)) {
      setAlert({
        visible: true,
        title: 'Already Connected',
        message: `Device ${device.mac} is already connected`,
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
      });
      return;
    }

    try {
      setConnecting(device.mac);
      // Add to database
      await pairDevice(profile.user_id, device.name ?? 'MY_SCALE', device.mac);
      // Add connection
      await connectDevice(device.mac);
      await refreshDevices();

      setAlert({
        visible: true,
        title: 'Success',
        message: `Connected to ${device.mac}`,
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
        cancelText: null,
      });
    } catch (error: any) {
      const errorMessage = error?.message || 'Unknown error occurred';
      setAlert({
        visible: true,
        title: 'Connection Failed',
        message: `Failed to connect to ${device.mac}\n\nError: ${errorMessage}`,
        confirmText: 'OK',
        onConfirm: () => setAlert({ ...alert, visible: false }),
        cancelText: null,
      });
      console.error('Connection error:', error);
    } finally {
      setConnecting(null);
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
      <View className="mb-4 flex-1 px-4">
        <ScrollView
          contentContainerStyle={{ flexGrow: 1 }}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }>
          <View className="flex-1">
            {/* Weight Data Display */}
            {/*{connectedDevices.length > 0 && (*/}
            {/*  <View className="mb-4 rounded-lg bg-purple-50 p-4">*/}
            {/*    <Text className="text-lg mb-2 font-semibold">*/}
            {/*      Weight Measurements*/}
            {/*    </Text>*/}
            {/*    {connectedDevices.map(device => {*/}
            {/*      const latestWeight = getLatestWeightForDevice(device.mac);*/}
            {/*      const deviceWeights = weightData*/}
            {/*        .filter(m => m.device.mac === device.mac)*/}
            {/*        .slice(-3); // Show last 3 measurements*/}

            {/*      return (*/}
            {/*        <View*/}
            {/*          key={device.mac}*/}
            {/*          className="mb-3 rounded-lg border bg-white p-3">*/}
            {/*          <Text className="text-sm mb-2 font-semibold">*/}
            {/*            Device: {device.mac}*/}
            {/*          </Text>*/}

            {/*          {latestWeight ? (*/}
            {/*            <View>*/}
            {/*              <Text className="text-2xl mb-1 font-bold text-purple-600">*/}
            {/*                {formatWeight(latestWeight.data.weight)} kg*/}
            {/*              </Text>*/}
            {/*              <Text className="text-xs mb-2 text-gray-500">*/}
            {/*                Last measurement:{' '}*/}
            {/*                {formatTime(latestWeight.data.timestamp)}*/}
            {/*                {latestWeight.data.isStabilized && ' (Stabilized)'}*/}
            {/*              </Text>*/}

            {/*              {deviceWeights.length > 1 && (*/}
            {/*                <View>*/}
            {/*                  <Text className="text-sm mb-1 font-medium">*/}
            {/*                    Recent measurements:*/}
            {/*                  </Text>*/}
            {/*                  {deviceWeights*/}
            {/*                    .reverse()*/}
            {/*                    .map((measurement, idx) => (*/}
            {/*                      <Text*/}
            {/*                        key={idx}*/}
            {/*                        className="text-xs text-gray-600">*/}
            {/*                        {formatWeight(measurement.data.weight)}kg at{' '}*/}
            {/*                        {formatTime(measurement.data.timestamp)}*/}
            {/*                      </Text>*/}
            {/*                    ))}*/}
            {/*                </View>*/}
            {/*              )}*/}
            {/*            </View>*/}
            {/*          ) : (*/}
            {/*            <Text className="italic text-gray-500">*/}
            {/*              No weight data yet - step on the scale to start*/}
            {/*              measuring*/}
            {/*            </Text>*/}
            {/*          )}*/}
            {/*        </View>*/}
            {/*      );*/}
            {/*    })}*/}
            {/*  </View>*/}
            {/*)}*/}

            {/* Scanned Devices */}
            <View className="flex-1">
              <Text className="mb-3 font-bodyBold text-secondary-500">
                Scanned Devices ({scannedDevices.length})
              </Text>
              {!bleEnabled ? (
                <View className="flex-1 items-center justify-center rounded-2xl bg-background-0 p-4">
                  <Text className="text-center text-primary-300">
                    Please enable Bluetooth to scan devices
                  </Text>
                </View>
              ) : scannedDevices.length === 0 ? (
                <View className="flex-1 items-center justify-center rounded-2xl bg-background-0 p-4">
                  <LoadingScreen text="Scanning for devices..." />
                </View>
              ) : (
                scannedDevices.map((device, index) => {
                  const isConnected = isDeviceConnected(device.mac);
                  const isConnectingThis = connecting === device.mac;

                  return (
                    <View
                      key={device.mac || index}
                      className={`mb-2 rounded-2xl bg-background-0 px-6 py-4`}>
                      <Text className="font-bodyBold text-body2 text-secondary-500">
                        Name: {device.name || 'Unknown Device'}
                      </Text>
                      <View className="mb-2 flex-row items-center justify-between">
                        <Text className="text-sm font-bodySemiBold text-primary-300">
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
            {/*<View className="mb-4">*/}
            {/*  <Text className="text-lg mb-2 font-semibold">*/}
            {/*    Connected Devices ({connectedDevices.length})*/}
            {/*  </Text>*/}
            {/*  {connectedDevices.length === 0 ? (*/}
            {/*    <View className="rounded-lg bg-gray-50 p-4">*/}
            {/*      <Text className="text-center italic text-gray-500">*/}
            {/*        No devices connected*/}
            {/*      </Text>*/}
            {/*    </View>*/}
            {/*  ) : (*/}
            {/*    connectedDevices.map((device, index) => {*/}
            {/*      const isDisconnectingThis = disconnecting === device.mac;*/}

            {/*      return (*/}
            {/*        <View*/}
            {/*          key={device.mac || index}*/}
            {/*          className="mb-2 rounded-lg border border-green-200 bg-green-50 p-4">*/}
            {/*          <View className="mb-2 flex-row items-center justify-between">*/}
            {/*            <Text className="font-bold">MAC: {device.mac}</Text>*/}
            {/*            <Text className="font-medium text-green-600">*/}
            {/*              🟢 Active*/}
            {/*            </Text>*/}
            {/*          </View>*/}

            {/*          <Text className="text-sm mb-3 text-gray-600">*/}
            {/*            Name: {device.name || 'Unknown Device'}*/}
            {/*          </Text>*/}

            {/*          <Text className="mb-3 text-center font-medium text-green-600">*/}
            {/*            📊 Ready to receive weight data*/}
            {/*          </Text>*/}
            {/*        </View>*/}
            {/*      );*/}
            {/*    })*/}
            {/*  )}*/}
            {/*</View>*/}
          </View>
        </ScrollView>
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
