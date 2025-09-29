import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import DateSelector from '@/components/DateSelector';
import { useEffect, useMemo, useState } from 'react';
import dayjs from 'dayjs';
import Header from '@/components/Header';
import { fetchAllWeightLogs } from '@/utils/body/api';
import { ScaleLog } from '@/types/database-types';

export default function AllLogs() {
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [data, setData] = useState<ScaleLog[] | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const result = await fetchAllWeightLogs();
        setData(result);
        console.log(result);
      } catch (error) {
        console.log('Error fetching all weight logs: ', error);
      }
    })();
  }, [selectedDate]);

  const filteredData = useMemo(() => {
    if (!data) return [];
    return data.filter(log =>
      dayjs(log.created_at).isSame(selectedDate, 'day'),
    );
  }, [data, selectedDate]);

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
        {filteredData.length > 0 ? (
          filteredData.map(log => (
            <View
              key={log.id}
              className="mb-2 rounded-xl bg-background-0 p-4 shadow-sm">
              <Text>Weight: {log.weight} kg</Text>
              <Text>BMI: {log.bmi}</Text>
              <Text>Body Fat: {log.body_fat}%</Text>
              <Text>Time: {dayjs(log.created_at).format('HH:mm')}</Text>
            </View>
          ))
        ) : (
          <Text className="mt-4 text-center text-primary-300">
            No logs for this date
          </Text>
        )}
      </View>
    </ThemedSafeAreaView>
  );
}
