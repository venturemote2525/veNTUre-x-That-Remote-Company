import Header from '@/components/Header';
import { Text, TextInput, ThemedSafeAreaView, View } from '@/components/Themed';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Image, Pressable, useColorScheme } from 'react-native';
import dayjs from 'dayjs';
import { useState, memo, useEffect } from 'react';
import DateTimePicker from '@react-native-community/datetimepicker';
import { toUpperCase } from '@/utils/formatString';
import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { PieChart, pieDataItem } from 'react-native-gifted-charts';
import { Colors } from '@/constants/Colors';
import LoadingScreen from '@/components/LoadingScreen';

const tempData = {
  date: '2025-08-22T16:36:41.702+00:00',
  name: '',
  meal: '',
  calories: 10,
  carbs: 5,
  protein: 7,
  fat: 8,
};

const mealColoursClasses: Record<string, string> = {
  breakfast: 'bg-[#8ecae6]',
  lunch: 'bg-[#ffb703]',
  dinner: 'bg-[#023e8a]',
};

export default function SummaryScreen() {
  const router = useRouter();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';
  const { image, meal: paramMeal } = useLocalSearchParams<{
    image: string;
    meal: string;
  }>();
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [data, setData] = useState({
    date: new Date(tempData.date),
    meal: paramMeal ?? '',
  });
  const [name, setName] = useState(tempData.name);
  const [selectedSlice, setSelectedSlice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const nutrients = [
    { key: 'Carbs', value: tempData.carbs, color: '#FFCA3A' },
    { key: 'Protein', value: tempData.protein, color: '#1982C4' },
    { key: 'Fat', value: tempData.fat, color: '#FF595E' },
  ];

  const decodedUri = decodeURIComponent(image);

  const handleDateChange = (event: any, date?: Date) => {
    if (date) {
      const updatedDate = new Date(data.date);
      updatedDate.setFullYear(
        date.getFullYear(),
        date.getMonth(),
        date.getDate(),
      );
      setData(prev => ({ ...prev, date: updatedDate }));
      setShowDatePicker(false);
      setShowTimePicker(true);
    } else {
      setShowDatePicker(false);
    }
  };

  const handleTimeChange = (event: any, time?: Date) => {
    if (time) {
      const updatedDate = new Date(data.date);
      updatedDate.setHours(
        time.getHours(),
        time.getMinutes(),
        time.getSeconds(),
      );
      setData(prev => ({ ...prev, date: updatedDate }));
    }
    setShowTimePicker(false);
  };

  const handleCancel = () => {
    router.replace('/(tabs)/logging');
  };

  const handleSave = async () => {
    try {
      // TODO: Save to supabase
    } finally {
      router.replace('/(tabs)/logging');
    }
  };

  useEffect(() => {
    try {
      setLoading(true);
      // TODO: Get analysis of photo
    } finally {
      setLoading(false);
    }
  }, []);

  if (loading)
    return (
      <ThemedSafeAreaView>
        <Header title="Food Summary" />
        <View className="flex-1">
          <LoadingScreen text="Analysing your photo" />
        </View>
      </ThemedSafeAreaView>
    );

  return (
    <ThemedSafeAreaView>
      <Header title="Food Summary" />
      <View className="flex-1 items-center gap-4 px-8 text-body1">
        <Pressable onPress={() => setShowDatePicker(true)}>
          <Text className="font-bodySemiBold text-primary-500">
            {dayjs(data.date).format('DD MMM YYYY HH:mm')}
          </Text>
        </Pressable>

        {image ? (
          <Image
            source={{ uri: decodedUri }}
            className="h-80 w-80 rounded-3xl"
          />
        ) : (
          <View className="h-80 w-80 items-center justify-center rounded-3xl border-2 border-secondary-500 bg-background-0">
            <Text className="font-bodyBold text-secondary-500">
              No photo found
            </Text>
          </View>
        )}

        <View className="w-full flex-row items-center justify-between">
          <FoodNameInput value={name} onChange={text => setName(text)} />
          <MemoizedMealDropdown
            meal={data.meal}
            setMeal={meal => setData(prev => ({ ...prev, meal }))}
          />
        </View>
        <View className="w-full flex-row items-center justify-start gap-8">
          <PieChart
            data={nutrients.map(n => ({
              value: n.value,
              text: n.key,
              color: n.color,
            }))}
            showText
            radius={90}
            labelsPosition="mid"
            font="Poppins-Medium"
            textColor="#1e1e1e"
            onPress={(item: pieDataItem) => setSelectedSlice(item.text ?? null)}
          />

          {/* Statistics */}
          <View className="gap-4">
            <Text className="font-bodyBold text-head2 text-secondary-500">
              {tempData.calories} kcal
            </Text>
            <View className="gap-1">
              {nutrients.map(n => (
                <Text
                  key={n.key}
                  className={`text-body2 ${
                    selectedSlice === n.key
                      ? 'font-bodyBold text-secondary-500'
                      : 'font-bodySemiBold text-primary-500'
                  }`}
                  style={{
                    color:
                      selectedSlice === n.key
                        ? n.color
                        : Colors[scheme].primary,
                  }}>
                  {n.value}g {n.key}
                </Text>
              ))}
            </View>
          </View>
        </View>
      </View>
      {/* Buttons */}
      <View className="flex-row gap-4 p-4">
        <Pressable
          onPress={handleCancel}
          className="button-rounded-tertiary flex-1">
          <Text className="font-bodyBold text-background-0">Cancel</Text>
        </Pressable>
        <Pressable onPress={handleSave} className="button-rounded flex-1">
          <Text className="font-bodyBold text-background-0">Save</Text>
        </Pressable>
      </View>
      {showDatePicker && (
        <DateTimePicker
          value={data.date}
          mode="date"
          display="spinner"
          maximumDate={new Date()}
          onChange={handleDateChange}
        />
      )}

      {showTimePicker && (
        <DateTimePicker
          value={data.date}
          mode="time"
          display="spinner"
          onChange={handleTimeChange}
        />
      )}
    </ThemedSafeAreaView>
  );
}

type FoodNameInputProps = {
  value: string;
  onChange: (text: string) => void;
};

const FoodNameInput = memo(({ value, onChange }: FoodNameInputProps) => {
  const [localValue, setLocalValue] = useState(value);

  const handleBlur = () => {
    onChange(localValue);
  };
  return (
    <TextInput
      value={localValue}
      onChangeText={setLocalValue}
      onBlur={handleBlur}
      placeholder="Food Name"
      multiline={true}
      maxLength={40}
      className="m-0 p-0 font-heading text-body2 text-primary-500"
      style={{ width: 200, paddingTop: 0, paddingBottom: 0 }}
    />
  );
});

FoodNameInput.displayName = 'FoodNameInput';

type MealDropdownProps = {
  meal: string;
  setMeal: (meal: string) => void;
};

const MealDropdown = ({ meal, setMeal }: MealDropdownProps) => (
  <CustomDropdown
    toggle={
      <Pressable
        className={`rounded-full px-4 py-2 ${mealColoursClasses[meal.toLowerCase()] || 'bg-secondary-500'}`}>
        <Text className="font-bodySemiBold text-background-0">
          {meal !== '' ? toUpperCase(meal) : 'Select Meal'}
        </Text>
      </Pressable>
    }
    menuClassName="gap-2 min-w-[120px] bg-background-0 p-2 rounded-2xl">
    {['breakfast', 'lunch', 'dinner'].map(m => (
      <DropdownItem
        key={m}
        label={toUpperCase(m)}
        onPress={() => setMeal(m)}
        itemClassName={`${mealColoursClasses[m]} rounded-xl px-2 py-1`}
        itemTextClassName="text-background-0"
      />
    ))}
  </CustomDropdown>
);

const MemoizedMealDropdown = memo(MealDropdown);
