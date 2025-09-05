import Header from '@/components/Header';
import TabToggle from '@/components/TabToggle';
import { View, Text, ThemedSafeAreaView } from '@/components/Themed';
import { useEffect, useState } from 'react';
import { Platform, Pressable } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import dayjs from 'dayjs';
import CustomWheelPicker from '@/components/CustomWheelPicker';
import {
  logDataManually,
  retrieveRecentHeight,
  retrieveRecentWeight,
} from '@/utils/body/api';
import { ManualLogEntry } from '@/types/database-types';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'expo-router';
import { CustomAlert } from '@/components/CustomAlert';
import { toUpperCase } from '@/utils/formatString';

const TABS = ['weight', 'height'];

export default function ManualLogging() {
  const { profile } = useAuth();
  const [screen, setScreen] = useState<(typeof TABS)[number]>('weight');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [date, setDate] = useState(new Date());
  const [weight, setWeight] = useState(60.0);
  const [height, setHeight] = useState(160.0);
  const [recentData, setRecentData] = useState({
    weight: 0,
    height: 0,
  });
  const router = useRouter();
  const [showAlert, setShowAlert] = useState(false);

  const handleDateChange = (event: any, date?: Date) => {
    setShowDatePicker(Platform.OS === 'ios'); // Keep open for IOS
    if (date) {
      setDate(date);
    }
  };

  // 20 to 200 kg
  const weightData = Array.from({ length: (480 - 30) * 4 + 1 }, (_, i) => {
    return +(20 + i * 0.1).toFixed(1);
  });
  // 120 to 280 cm
  const heightData = Array.from({ length: (430 - 30) * 4 + 1 }, (_, i) => {
    return +(120 + i * 0.1).toFixed(1);
  });

  useEffect(() => {
    (async () => {
      try {
        const weight = await retrieveRecentWeight();
        const height = await retrieveRecentHeight();
        setRecentData({ weight, height });
        // Optionally, set initial picker values if available
        if (weight) setWeight(weight);
        if (height) setHeight(height);
      } catch (error) {
        console.log('Error fetching weight and height: ', error);
      }
    })();
  }, []);

  const handleConfirm = async () => {
    if (!profile) return;
    try {
      const log: ManualLogEntry = {
        logged_at: date.toISOString(),
        user_id: profile.user_id,
      };

      if (screen === 'weight') {
        log.weight = weight;
      } else if (screen === 'height') {
        log.height = height;
      }

      await logDataManually(log);
      setShowAlert(true);
    } catch (error) {
      console.log('Manual logging error: ', error);
    }
  };

  function renderScreen() {
    switch (screen) {
      case 'weight':
        return (
          <View className="items-center justify-center gap-4">
            {recentData.weight ? (
              <Text className="text py-5">
                Previous weight: {recentData.weight.toFixed(1)} kg
              </Text>
            ) : null}
            <View className="flex-row items-center gap-4">
              <CustomWheelPicker
                data={weightData}
                selectedIndex={weightData.findIndex(w => w === weight)}
                onSelect={setWeight}
                textSize={36}
              />
              <Text className="font-bodySemiBold text-head1 text-primary-500">
                kg
              </Text>
            </View>
          </View>
        );
      case 'height':
        return (
          <View className="items-center gap-4">
            {recentData.height ? (
              <Text className="text py-5">
                Previous height: {recentData.height.toFixed(1)} cm
              </Text>
            ) : null}
            <View className="flex-row items-center gap-4">
              <CustomWheelPicker
                data={heightData}
                selectedIndex={heightData.findIndex(h => h === height)}
                onSelect={setHeight}
                textSize={36}
              />
              <Text className="font-bodySemiBold text-head1 text-primary-500">
                cm
              </Text>
            </View>
          </View>
        );
    }
  }

  return (
    <ThemedSafeAreaView>
      <Header title="Manual Log" />
      {/* Date Selector */}
      <View className="items-center">
        <Pressable
          className="rounded-full border-2 border-secondary-500 px-4 py-2"
          onPress={() => setShowDatePicker(true)}>
          <Text className="font-bodySemiBold text-secondary-500">
            {dayjs(date).format('MMM D, YYYY')}
          </Text>
        </Pressable>
      </View>
      <TabToggle tabs={TABS} selectedTab={screen} onTabChange={setScreen} />
      <View className="flex-1 gap-4 p-4">
        <View className="flex-1 items-center justify-center">
          {renderScreen()}
        </View>
        <View>
          <Pressable className="button-rounded" onPress={handleConfirm}>
            <Text className="button-rounded-text">Confirm</Text>
          </Pressable>
        </View>
      </View>

      {showDatePicker && (
        <DateTimePicker
          value={date}
          mode="date"
          display="spinner"
          maximumDate={new Date()}
          onChange={handleDateChange}
        />
      )}
      <CustomAlert
        visible={showAlert}
        title={`${toUpperCase(screen)} logged!`}
        onConfirm={() => router.back()}
      />
    </ThemedSafeAreaView>
  );
}
