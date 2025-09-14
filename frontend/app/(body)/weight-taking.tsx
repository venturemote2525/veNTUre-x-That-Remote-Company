import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import React, { useEffect } from 'react';
import Header from '@/components/Header';
import { useUserInfo } from '@/context/UserInfoContext';
import { NativeModules } from 'react-native';

const { ICDeviceModule } = NativeModules;

export default function WeightTaking() {
  const { connectedDevices, getLatestWeightForDevice, weightData } =
    useICDevice();
  const { userInfo, loading: userInfoLoading } = useUserInfo();

  // Send user info to native device module when it changes
  useEffect(() => {
    const updateDeviceUserInfo = async () => {
      if (userInfo && ICDeviceModule && connectedDevices.length > 0) {
        // Send user info to each connected device
        for (const device of connectedDevices) {
          try {
            console.log(`Sending user info to device ${device.mac}:`, userInfo);
            await ICDeviceModule.setUserInfo(device.mac, {
              name: userInfo.name || "User", // Add name field
              age: userInfo.age,
              height: userInfo.height,
              gender: userInfo.sex === 2 ? "FEMALE" : "MALE" // Convert numeric to string
            });
            console.log(`User info sent to device ${device.mac} successfully`);
          } catch (error) {
            console.error(`Failed to send user info to device ${device.mac}:`, error);
          }
        }
      }
    };

    updateDeviceUserInfo();
  }, [userInfo, connectedDevices]);

  const formatWeight = (weight: number | undefined | null): string => {
    if (weight == null) return 'N/A';
    return weight.toFixed(2);
  };

  const formatTime = (timestamp: number): string =>
    new Date(timestamp).toLocaleTimeString();

  const getSexString = (sex: number): string => {
    return sex === 2 ? 'Female' : 'Male';
  };

  const getPeopleTypeString = (peopleType: number): string => {
    return peopleType === 1 ? 'Athlete' : 'Normal';
  };

  return (
    <ThemedSafeAreaView>
      <Header title="Weight Taking" />

      {/* User Information Section */}
      {!userInfoLoading && (
        <View className="mb-4 rounded-lg bg-blue-50 p-4">
          <Text className="text-lg mb-3 font-semibold text-blue-800">
            User Profile
          </Text>

          {userInfo ? (
            <View className="space-y-2">
              <View className="flex-row justify-between items-center">
                <Text className="text-sm font-medium text-gray-700">Age:</Text>
                <Text className="text-sm font-semibold text-blue-600">
                  {userInfo.age} years
                </Text>
              </View>

              <View className="flex-row justify-between items-center">
                <Text className="text-sm font-medium text-gray-700">Height:</Text>
                <Text className="text-sm font-semibold text-blue-600">
                  {userInfo.height} cm
                </Text>
              </View>

              <View className="flex-row justify-between items-center">
                <Text className="text-sm font-medium text-gray-700">Sex:</Text>
                <Text className="text-sm font-semibold text-blue-600">
                  {getSexString(userInfo.sex)}
                </Text>
              </View>

              <View className="flex-row justify-between items-center">
                <Text className="text-sm font-medium text-gray-700">Type:</Text>
                <Text className="text-sm font-semibold text-blue-600">
                  {getPeopleTypeString(userInfo.peopleType)}
                </Text>
              </View>

              {userInfo.status === 'success' && (
                <Text className="text-xs text-green-600 italic">
                  ✓ Profile loaded from user account
                </Text>
              )}

              {userInfo.status === 'default' && (
                <Text className="text-xs text-orange-600 italic">
                  ⚠ Using default values
                </Text>
              )}
            </View>
          ) : (
            <Text className="italic text-gray-500">
              No user profile available
            </Text>
          )}
        </View>
      )}

      {userInfoLoading && (
        <View className="mb-4 rounded-lg bg-gray-50 p-4">
          <Text className="text-gray-500">Loading user profile...</Text>
        </View>
      )}

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
                key={`${device.mac} - ${Date.now()}`}
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

                    {/* Enhanced BMI Display with user height */}
                    {latestWeight.data.BMI != null ? (
                      <Text className="text-sm mb-1 font-medium">
                        BMI: {formatWeight(latestWeight.data.BMI)} kg/m²
                      </Text>
                      )
                    : null}

                    {latestWeight.data.bodyFat != null && (
                      <Text className="text-sm font-medium">
                        Body Fat: {formatWeight(latestWeight.data.bodyFat)}%
                      </Text>
                    )}

                    {/* Additional health insights based on user info */}
                    {userInfo && latestWeight.data.weight && (
                      <View className="mt-2 p-2 bg-gray-50 rounded">
                        <Text className="text-xs text-gray-600">
                          Weight for {getSexString(userInfo.sex).toLowerCase()},
                          age {userInfo.age}, height {userInfo.height}cm
                        </Text>
                        {userInfo.peopleType === 1 && (
                          <Text className="text-xs text-blue-600">
                            ⚡ Athlete profile active
                          </Text>
                        )}
                      </View>
                    )}

                    {deviceWeights.length > 1 && (
                      <View className="mt-3">
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

      {connectedDevices.length === 0 && (
        <View className="rounded-lg bg-gray-50 p-4">
          <Text className="text-center text-gray-500">
            No connected devices. Please connect a weight scale to start measuring.
          </Text>
        </View>
      )}
    </ThemedSafeAreaView>
  );
}