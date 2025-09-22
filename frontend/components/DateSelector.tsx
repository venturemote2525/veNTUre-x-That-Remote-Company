import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { faCaretLeft, faCaretRight } from '@fortawesome/free-solid-svg-icons';
import { View, Text } from '@/components/Themed';
import { Pressable, useColorScheme } from 'react-native';
import { useState } from 'react';
import DateTimePicker from '@react-native-community/datetimepicker';
import dayjs from 'dayjs';
import { Colors } from '@/constants/Colors';

type DateSelectorProps = {
  selectedDate: dayjs.Dayjs;
  onDateChange: (date: dayjs.Dayjs) => void;
};

export default function DateSelector({
  selectedDate,
  onDateChange,
}: DateSelectorProps) {
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';
  const today = dayjs();
  const [showPicker, setShowPicker] = useState(false);

  const handlePrev = () => {
    const newDate = selectedDate.subtract(1, 'day');
    onDateChange(newDate);
  };

  const handleNext = () => {
    const next = selectedDate.add(1, 'day');
    onDateChange(next.isAfter(today) ? today : next);
  };

  const handleDateChange = (_event: any, date?: Date) => {
    setShowPicker(false);
    if (date) {
      const newDate = dayjs(date);
      onDateChange(newDate.isAfter(today) ? today : newDate);
    }
  };

  return (
    <View className="w-full flex-row items-center justify-center gap-4 py-3">
      <Pressable onPress={handlePrev}>
        <FontAwesomeIcon
          icon={faCaretLeft}
          size={24}
          color={Colors[scheme].primary}
        />
      </Pressable>

      <Pressable onPress={() => setShowPicker(true)}>
        <Text className="font-bodySemiBold text-body2 text-primary-500">
          {selectedDate.isSame(today, 'day')
            ? 'Today'
            : selectedDate.format('DD MMM YYYY')}
        </Text>
      </Pressable>

      <Pressable
        onPress={handleNext}
        disabled={selectedDate.isSame(today, 'day')}>
        <FontAwesomeIcon
          icon={faCaretRight}
          size={24}
          color={
            selectedDate.isSame(today, 'day')
              ? '#6c757d'
              : Colors[scheme].primary
          }
        />
      </Pressable>

      {showPicker && (
        <DateTimePicker
          value={selectedDate.toDate()}
          mode="date"
          maximumDate={today.toDate()}
          onChange={handleDateChange}
        />
      )}
    </View>
  );
}
