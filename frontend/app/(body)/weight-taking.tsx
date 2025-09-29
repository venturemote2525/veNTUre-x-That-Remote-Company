import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { useICDevice } from '@/context/ICDeviceContext';
import React, { useEffect, useState } from 'react';
import Header from '@/components/Header';
import { Pressable, NativeModules } from 'react-native';
import { AlertState, ScaleLogEntry } from '@/types/database-types';
import { useAuth } from '@/context/AuthContext';
import { logScaleData } from '@/utils/body/api';
import { CustomAlert } from '@/components/CustomAlert';
import { useRouter } from 'expo-router';
import { useUserInfo } from '@/context/UserInfoContext';
import { calculateAge } from '@/utils/helpers';

const { ICDeviceModule } = NativeModules;

export default function WeightTaking() {
  const { connectedDevices, getLatestWeightForDevice, pairedDevices } =
    useICDevice();
  const { profile } = useAuth();
  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [userInfoSent, setUserInfoSent] = useState(false);

  const formatWeight = (weight: number | undefined | null): string => {
    if (weight == null) return 'N/A';
    return weight.toFixed(2);
  };

  const device = connectedDevices[0];
  const latestWeight = device ? getLatestWeightForDevice(device.mac) : null;
  const dbDevice = device
    ? pairedDevices.find(d => d.mac === device.mac)
    : null;

  // Extract the updateDeviceUserInfo function
  const updateDeviceUserInfo = async () => {
    if (profile && ICDeviceModule && device) {
      try {
        const userProfile = Array.isArray(profile) ? profile[0] : profile;
        console.log(`Sending user info to device ${device.mac}`, userProfile);
        await ICDeviceModule.setUserInfo(device.mac, {
          name: userProfile.username || 'User',
          age: calculateAge(userProfile.dob) || 25,
          height: userProfile.height || 170,
          gender:
            userProfile.gender?.toUpperCase() === 'FEMALE' ? 'FEMALE' : 'MALE',
        });
        console.log(`User info sent successfully`);
        setUserInfoSent(true);
      } catch (error) {
        console.error(`Failed to send user info:`, error);
      }
    }
  };

  // Send user info when weight is stabilized
  useEffect(() => {
    if (latestWeight?.data.isStabilized && !userInfoSent) {
      updateDeviceUserInfo();
    }
  }, [latestWeight?.data.isStabilized, userInfoSent]);

  // Reset userInfoSent when device changes or when no weight data
  useEffect(() => {
    setUserInfoSent(false);
  }, [device?.mac, latestWeight?.data.weight]);

  // Progress Bar Timer
  useEffect(() => {
    let start: number | null = null;
    let frame: number;
    const animate = (timestamp: number) => {
      if (!start) start = timestamp;
      const elapsed = timestamp - start;
      const progressValue = Math.min(elapsed / 5000, 1);
      setProgress(progressValue);
      if (progressValue < 1) {
        frame = requestAnimationFrame(animate);
      }
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, []);

  const logWeight = async () => {
    if (!profile || !latestWeight) return;
    try {
      const userProfile = Array.isArray(profile) ? profile[0] : profile;
      const log: ScaleLogEntry = {
        user_id: userProfile.user_id,
        weight: latestWeight.data.weight || 0,
        bmi: latestWeight.data.BMI || 0,
        body_fat: latestWeight.data.bodyFat || 0,
      };

      console.log('logWeight', log);
      await logScaleData(log);

      setAlert({
        visible: true,
        title: 'Weight logged!',
        message: '',
        onConfirm: () => {
          setAlert({ ...alert, visible: false });
          router.back();
        },
      });
    } catch (error: any) {
      console.error('Error logging weight:', error);
      setAlert({
        title: 'Error logging weight',
        message: error.message || 'Unknown error occurred',
        visible: true,
      });
    }
  };

  const isReady = progress >= 1 && latestWeight?.data.isStabilized;

  return (
    <ThemedSafeAreaView>
      <Header title="Weight Taking" />
      <View className="flex-1 items-center p-4">
        {device ? (
          <View className="flex-1 items-center">
            <Text className="font-bodySemiBold text-body2 text-primary-300">
              Device: {dbDevice ? dbDevice.name : device.mac}
            </Text>

            {latestWeight ? (
              <View className="flex-1 items-center justify-center">
                <Text className="font-bodyBold text-title text-secondary-500">
                  {formatWeight(latestWeight.data.weight)} kg
                </Text>

                {/* BMI Display */}
                {latestWeight.data.BMI != null && (
                  <Text className="text-sm mt-1 text-gray-500">
                    BMI: {formatWeight(latestWeight.data.BMI)} kg/m²
                  </Text>
                )}

                {/* Body Fat Display */}
                {latestWeight.data.bodyFat != null && (
                  <Text className="text-sm text-gray-500">
                    Body Fat: {formatWeight(latestWeight.data.bodyFat)}%
                  </Text>
                )}

                {/* User Info Status */}
                {latestWeight.data.isStabilized && userInfoSent && (
                  <Text className="text-xs mt-2 text-green-500">
                    User info sent to device ✓
                  </Text>
                )}
              </View>
            ) : (
              <Text className="italic text-primary-300">
                No weight data yet - step on the scale to start measuring
              </Text>
            )}
          </View>
        ) : (
          <Text className="italic text-primary-300">No device connected</Text>
        )}

        {/* Progress Bar */}
        <View className="mx-1 my-4 h-3 w-full rounded-full bg-background-300">
          <View
            className="h-3 rounded-full bg-secondary-500"
            style={{ width: `${progress * 100}%` }}
          />
        </View>

        <Pressable
          disabled={!isReady}
          className={`w-full items-center rounded-full py-3 ${
            isReady ? 'bg-secondary-500' : 'bg-background-400'
          }`}
          onPress={logWeight}>
          <Text className="button-rounded-text">Log</Text>
        </Pressable>
      </View>

      <CustomAlert
        visible={alert.visible}
        title={alert.title}
        message={alert.message}
        onConfirm={alert.onConfirm ?? (() => {})}
      />
    </ThemedSafeAreaView>
  );
}
