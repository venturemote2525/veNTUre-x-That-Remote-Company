import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import React from 'react';
import Header from '@/components/Header';

export default function WeightTaking() {
  const {
    connectedDevices,
    getLatestWeightForDevice,
    weightData,
    pairedDevices,
  } = useICDevice();

  const formatWeight = (weight: number): string => weight.toFixed(2);
  const formatTime = (timestamp: number): string =>
    new Date(timestamp).toLocaleTimeString();

  return (
    <ThemedSafeAreaView>
      <Header title="Weight Taking" />

      <View className="flex-1 items-center px-4">
        {connectedDevices.length > 0 &&
          connectedDevices.map(device => {
            const latestWeight = getLatestWeightForDevice(device.mac);
            const deviceWeights = weightData
              .filter(m => m.device.mac === device.mac)
              .slice(-3); // Show last 3 measurements
            const dbDevice = pairedDevices.find(d => d.mac === device.mac);
            return (
              <View key={device.mac} className="gap-4">
                <Text className="font-bodySemiBold text-body2 text-primary-500">
                  Device: {dbDevice ? dbDevice.name : device.mac}
                </Text>

                {latestWeight ? (
                  <View>
                    <Text className="mb-1 font-bodyBold text-title text-secondary-500">
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
                        {deviceWeights
                          .slice()
                          .reverse()
                          .map((measurement, idx) => (
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
                    No weight data yet - step on the scale to start measuring
                  </Text>
                )}
              </View>
            );
          })}
      </View>
    </ThemedSafeAreaView>
  );
}
