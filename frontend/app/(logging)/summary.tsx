import Header from '@/components/Header';
import { Text, TextInput, ThemedSafeAreaView, View } from '@/components/Themed';
import { useLocalSearchParams, useNavigation, useRouter } from 'expo-router';
import { Image, Pressable, useColorScheme } from 'react-native';
import dayjs from 'dayjs';

import { useState, memo, useEffect } from 'react';
import DateTimePicker from '@react-native-community/datetimepicker';
import { toUpperCase } from '@/utils/formatString';
import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { PieChart, pieDataItem } from 'react-native-gifted-charts';
import { Colors } from '@/constants/Colors';
import LoadingScreen from '@/components/LoadingScreen';
import { deleteMeal, retrieveMeal, updateMeal } from '@/utils/food/api';
import { AlertState, Meal } from '@/types/database-types';
import { CustomAlert } from '@/components/CustomAlert';
import { useFocusEffect } from '@react-navigation/native';
import { BackHandler } from 'react-native';

const mealColoursClasses: Record<string, string> = {
  breakfast: 'bg-[#8ecae6]',
  lunch: 'bg-[#ffb703]',
  dinner: 'bg-[#023e8a]',
  morning_snack: 'bg-[#8ecae6]',
  afternoon_snack: 'bg-[#ffb703]',
  night_snack: 'bg-[#023e8a]',
};

export default function SummaryScreen() {
  const router = useRouter();
  const navigation = useNavigation();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';
  const { mealId, meal: paramMeal, type } = useLocalSearchParams<{
    mealId: string;
    meal: string;
    type: string;
  }>();
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showTimePicker, setShowTimePicker] = useState(false);
  const [data, setData] = useState({
    date: new Date(),
    meal: paramMeal ?? '',
  });
  const [name, setName] = useState('');
  const [selectedSlice, setSelectedSlice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [mealData, setMealData] = useState<Meal | null>(null);
  const [nutrients, setNutrients] = useState([
    { key: 'Carbs', value: 0, color: '#FFCA3A' },
    { key: 'Protein', value: 0, color: '#1982C4' },
    { key: 'Fat', value: 0, color: '#FF595E' },
  ]);
  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  })

  useEffect(() => {
    const fetchMeal = async () => {
      const data = await retrieveMeal(mealId);
      setMealData(data);
      setData(prev => ({
        ...prev,
        date: new Date(data.date),
        meal: type === 'history' ? data.meal.toLowerCase() : prev.meal,
      }));
      setName(data.name)
      setNutrients([
        { key: 'Carbs', value: data.carbs, color: '#FFCA3A' },
        { key: 'Protein', value: data.protein, color: '#1982C4' },
        { key: 'Fat', value: data.fat, color: '#FF595E' },
      ]);
      console.log('Meal data: ', data);
    }
    try {
      setLoading(true);
      fetchMeal();
      setLoading(false);
    } catch (error) {
      console.log('Fetch meal error: ', error);
    }
  }, []);

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
    setAlert({
      visible: true,
      title: type === 'log' ? 'Discard meal log?' : 'Discard changes?',
      message: type === 'log' ? `Are you sure you want to discard ${name}` : 'Are you sure you want to discard changes',
      confirmText: 'Yes',
      onConfirm: () => {handleConfirmCancel()},
      cancelText: 'No',
      onCancel: () => setAlert({ ...alert, visible: false }),
    })
  }

  const handleConfirmCancel = async () => {
    if (type === 'history') {
      navigation.goBack();
      return;
    }
    try {
      await deleteMeal(mealId);
      setAlert({ ...alert, visible: false });
      navigation.goBack();
    } catch (error) {
      console.log(error);
    }
  };

  const handleSave = async () => {
    if (!mealData) return
    if (data.meal === '') {
      setAlert({
        visible: true,
        title: 'Missing info',
        message: 'Please select meal',
        onConfirm: () => {setAlert(() => ({ ...alert, visible: false })) }
      })
      return;
    }
    setLoading(true);
    try {
      await updateMeal(mealId, {
        name: name,
        date: data.date.toISOString(),
        meal: data.meal.toUpperCase(),
      });
      router.replace('/(tabs)/food');
    } catch (error) {
      console.log('Update meal error: ', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const subscription = BackHandler.addEventListener(
      'hardwareBackPress',
      () => {
        handleCancel()
        return true
      },
    )
    return () => subscription.remove()
  }, [])

  if (loading)
    return (
      <ThemedSafeAreaView>
        <Header title="Food Summary" />
        <View className="flex-1">
          <LoadingScreen text="Loading" />
        </View>
      </ThemedSafeAreaView>
    );

  return (
    <ThemedSafeAreaView>
      <Header title="Food Summary" onBackPress={handleCancel} />
      <View className="flex-1 items-center gap-4 px-8 text-body1">
        <Pressable onPress={() => setShowDatePicker(true)}>
          <Text className="font-bodySemiBold text-primary-500">
            {dayjs(data.date).format('DD MMM YYYY HH:mm')}
          </Text>
        </Pressable>

        {mealData?.image_url ? (
          <Image
            source={{ uri: mealData?.image_url }}
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
          {nutrients.reduce((sum, n) => sum + n.value, 0) > 0 ? (
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
          ) : (
              <Text className="text-secondary-500">No nutrients data</Text>
          )}

          {/* Statistics */}
          <View className="gap-4">
            <Text className="font-bodyBold text-head2 text-secondary-500">
              {mealData?.calories ?? 0} kcal
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

type FoodNameInputProps = {
  value: string;
  onChange: (text: string) => void;
};

const FoodNameInput = memo(({ value, onChange }: FoodNameInputProps) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

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
      style={{ width: '50%', paddingTop: 0, paddingBottom: 0 }}
      submitBehavior={'blurAndSubmit'}
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
        className={`w-full rounded-full px-4 py-2 ${mealColoursClasses[meal.toLowerCase()] || 'bg-secondary-500'}`}>
        <Text className="font-bodySemiBold text-background-0">
          {meal !== '' ? toUpperCase(meal) : 'Select Meal'}
        </Text>
      </Pressable>
    }
    gap={8}
    menuClassName="gap-2 min-w-40 bg-background-0 p-2 rounded-2xl">
    {Object.keys(mealColoursClasses).map(m => (
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
