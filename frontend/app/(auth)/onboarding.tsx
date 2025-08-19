import { Text, TextInput, ThemedSafeAreaView, View } from '@/components/Themed';
import { useState } from 'react';
import { Pressable } from 'react-native';

export default function Onboarding() {
  const [loading, setLoading] = useState(false);
  const [fields, setFields] = useState({
    gender: '',
    name: '',
    dob: '',
    height: '',
  });

  const handleConfirm = async () => {
    try {
      setLoading(true);
      // TODO: Update user details in supabase
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedSafeAreaView className="px-4">
      <View className="mb-8 items-center gap-1 bg-slate-600 px-2">
        <Text className="text-head2 font-heading">Welcome to HealthSync</Text>
        <Text className="text-primary-200">Please enter your details</Text>
      </View>
      {/* Name */}
      <View className="">
        <Text className="px-2 text-sm text-primary-300">Your Name</Text>
        <TextInput
          placeholder="Name"
          onChangeText={text => setFields({ ...fields, name: text })}
          className="h-14 items-center rounded-2xl bg-background-0 px-4"
        />
      </View>
      <Pressable
        onPress={handleConfirm}
        className="items-center rounded-2xl bg-secondary-500 py-3">
        <Text className="font-bodyBold text-xl text-background-500">
          Confirm
        </Text>
      </Pressable>
    </ThemedSafeAreaView>
  );
}
