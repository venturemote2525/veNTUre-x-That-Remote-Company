import { Text, TextInput, ThemedSafeAreaView, View } from '@/components/Themed';
import { useState } from 'react';
import { Platform, Pressable } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';

export default function Onboarding() {
  const [loading, setLoading] = useState(false);
  const genders = ['Female', 'Male'];
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [fields, setFields] = useState({
    gender: '',
    name: '',
    dob: '',
    height: '',
  });
  const [error, setError] = useState('');
  const [selectedDate, setSelectedDate] = useState(new Date());

  const handleConfirm = async () => {
    setError('');
    // Check if any fields empty
    const emptyFields = Object.entries(fields).filter(
      ([key, value]) => !value.trim(),
    );
    if (emptyFields.length > 0) {
      setError('Please fill in all details');
      return;
    }
    try {
      setLoading(true);
      // TODO: Update user details in supabase
    } finally {
      setLoading(false);
    }
  };

  const handleDateChange = (event: any, date?: Date) => {
    setShowDatePicker(Platform.OS === 'ios'); // Keep open for IOS
    if (date) {
      setSelectedDate(date);
      setFields({
        ...fields,
        dob: date.toLocaleDateString(),
      });
    }
  };

  return (
    <ThemedSafeAreaView className="px-4">
      <Text className="px-2 text-sm text-primary-300">{fields.name}</Text>
      <Text className="px-2 text-sm text-primary-300">{fields.dob}</Text>
      <Text className="px-2 text-sm text-primary-300">{fields.gender}</Text>
      <Text className="px-2 text-sm text-primary-300">{fields.height}</Text>
      <View className="items-center gap-1 py-12">
        <Text className="text-head2 font-heading">Welcome to HealthSync</Text>
        <Text className="text-primary-200">Please enter your details</Text>
      </View>

      <View className="flex-1 gap-4">
        {/* Gender */}
        <View className="">
          <Text className="px-2 text-sm text-primary-300">Your gender</Text>
          <View className="flex-row justify-center gap-8">
            {genders.map(gender => (
              <Pressable
                key={gender}
                onPress={() => setFields({ ...fields, gender: gender })}
                className={`flex-1 items-center rounded-2xl p-4 ${fields.gender === gender ? 'bg-secondary-500' : 'bg-background-0'}`}>
                <Text
                  className={`${fields.gender === gender ? 'font-bodyBold text-background-0' : ''}`}>
                  {gender}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
        {/* Name */}
        <View className="">
          <Text className="px-2 text-sm text-primary-300">Your name</Text>
          <TextInput
            placeholder="Name"
            onChangeText={text => setFields({ ...fields, name: text })}
            className="h-14 items-center rounded-2xl bg-background-0 px-4"
          />
        </View>
        {/* Height */}
        <View className="">
          <Text className="px-2 text-sm text-primary-300">Your height</Text>
          <View className="h-14 flex-row items-center rounded-2xl bg-background-0 px-4">
            <TextInput
              keyboardType="number-pad"
              placeholder="Height"
              onChangeText={text => setFields({ ...fields, height: text })}
              className="flex-1"
            />
            <Text>cm</Text>
          </View>
        </View>
        {/* DOB */}
        <View className="">
          <Text className="px-2 text-sm text-primary-300">
            Your date of birth
          </Text>
          <Pressable
            className="h-14 justify-center rounded-2xl bg-background-0 px-4"
            onPress={() => setShowDatePicker(true)}>
            <Text className={`${fields.dob === '' ? 'text-[#5d5d5d]' : ''}`}>
              {fields.dob || 'Select date'}
            </Text>
          </Pressable>
        </View>
      </View>
      {error && <Text className="text-center text-error-500">{error}</Text>}
      <Pressable
        onPress={handleConfirm}
        className="mb-4 items-center rounded-2xl bg-secondary-500 py-3">
        <Text className="font-bodyBold text-xl text-background-500">
          Confirm
        </Text>
      </Pressable>

      {showDatePicker && (
        <DateTimePicker
          value={selectedDate}
          mode="date"
          display="spinner"
          maximumDate={new Date()}
          onChange={handleDateChange}
        />
      )}
    </ThemedSafeAreaView>
  );
}
