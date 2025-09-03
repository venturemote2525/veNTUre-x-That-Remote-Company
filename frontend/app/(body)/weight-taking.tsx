import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import React from 'react';
import Header from '@/components/Header';

export default function WeightTaking() {
  const { connectedDevices, getLatestWeightForDevice, weightData } =
    useICDevice();

  const formatWeight = (weight: number): string => weight.toFixed(2);
  const formatTime = (timestamp: number): string =>
    new Date(timestamp).toLocaleTimeString();

  return (
    <ThemedSafeAreaView>
      <Header title="Weight Taking" />
      <Text>Weight</Text>

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
      )}
    </ThemedSafeAreaView>
  );
}
