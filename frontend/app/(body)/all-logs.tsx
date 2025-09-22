import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import DateSelector from '@/components/DateSelector';
import { useState } from 'react';
import dayjs from 'dayjs';
import Header from '@/components/Header';

export default function AllLogs() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  return (
    <ThemedSafeAreaView>
      <Header title="All Logs" />

      <View className="flex-1 px-4">
        {/* Date Selector */}
        <DateSelector
          selectedDate={selectedDate}
          onDateChange={setSelectedDate}
        />
        <Text>all logs</Text>
      </View>
    </ThemedSafeAreaView>
  );
}
