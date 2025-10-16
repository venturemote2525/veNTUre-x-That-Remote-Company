import { Text, TextInput, ThemedSafeAreaView, View } from '@/components/Themed';
import { useState } from 'react';
import { Platform, Pressable } from 'react-native';
import DateTimePicker from '@react-native-community/datetimepicker';
import { useRouter } from 'expo-router';
import { useAuth } from '@/context/AuthContext';
import { createProfile } from '@/utils/auth/api';

export default function Onboarding() {
  const router = useRouter();
  const { user, refreshProfile } = useAuth();
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
    if (!user) return;
    console.log('userid: ', user?.id);
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
      if (!user) return;
      setLoading(true);
      // Insert user details in supabase
      const profile = {
        user_id: user?.id,
        username: fields.name,
        gender: fields.gender.toUpperCase(),
        height: Number(fields.height),
        dob: selectedDate.toISOString().split('T')[0],
      };
      await createProfile(profile);
      refreshProfile();
      router.replace('/(tabs)/home');
    } catch (error) {
      console.log('Onboarding error: ', error);
    } finally {
      setLoading(false);
      router.replace('/(tabs)/home');
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
      <View className="items-center gap-1 py-12">
        <Text className="font-heading text-head2 text-primary-500">
          Welcome to STRIDE
        </Text>
        <Text className="text-primary-200">Please enter your details</Text>
      </View>

      <View className="flex-1 gap-4">
        {/* Gender */}
        <View className="">
          <Text className="text-sm px-2 text-primary-300">Your gender</Text>
          <View className="flex-row justify-center gap-8">
            {genders.map(gender => (
              <Pressable
                key={gender}
                onPress={() => setFields({ ...fields, gender: gender })}
                className={`flex-1 items-center rounded-2xl p-4 ${fields.gender === gender ? 'bg-secondary-500' : 'bg-background-0'}`}>
                <Text
                  className={`${fields.gender === gender ? 'font-bodyBold text-background-0' : 'text-primary-500'}`}>
                  {gender}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>
        {/* Name */}
        <View className="">
          <Text className="text-sm px-2 text-primary-300">Your name</Text>
          <TextInput
            placeholder="Name"
            onChangeText={text => setFields({ ...fields, name: text })}
            className="h-14 items-center rounded-2xl bg-background-0 px-4"
          />
        </View>
        {/* Height */}
        <View className="">
          <Text className="text-sm px-2 text-primary-300">Your height</Text>
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
          <Text className="text-sm px-2 text-primary-300">
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
        <Text className="text-xl font-bodyBold text-background-500">
          {loading ? 'Creating your profile...' : 'Confirm'}
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
