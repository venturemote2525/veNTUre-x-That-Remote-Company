import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'expo-router';
import { useEffect, useState } from 'react';
import { Pressable } from 'react-native';

export default function ProfileScreen() {
  const router = useRouter();
  const { profile } = useAuth();
  const [userEmail, setUserEmail] = useState<string>('');
  const [createdAt, setCreatedAt] = useState<string>('');

  // Fetch user info from Supabase
  useEffect(() => {
    const fetchUserInfo = async () => {
      if (!profile?.id) return;

      const { data, error } = await supabase
        .from('profiles') // Assuming you have a 'profiles' table with created_at
        .select('email, created_at')
        .eq('id', profile.id)
        .single();

      if (error) {
        console.log('Error fetching user info:', error);
      } else if (data) {
        setUserEmail(data.email);
        setCreatedAt(new Date(data.created_at).toLocaleDateString());
      }
    };

    fetchUserInfo();
  }, [profile]);

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      console.log(error);
      router.replace('/');
    } catch (error) {
      console.log('Error signing out: ', error);
    }
  };

  return (
    <ThemedSafeAreaView>
      <View className="flex-1 justify-between px-4 py-8">
        <View className="space-y-2">
          <Text className="text-primary-500 font-bodyBold text-body1">
            {profile?.username}
          </Text>
          <Text className="font-body text-body2 text-black">
            Email: {userEmail || 'Loading...'}
          </Text>
          <Text className="font-body text-body2 text-black">
            Member since: {createdAt || 'Loading...'}
          </Text>
        </View>

        <Pressable onPress={handleLogout} className="button-rounded py-3 px-4">
          <Text className="font-bodyBold text-background-0 text-center">Log out</Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
