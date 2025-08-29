import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useAuth } from '@/context/AuthContext';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'expo-router';
import { Pressable, Image } from 'react-native';
import { Link } from 'expo-router';

export default function ProfileScreen() {
  const router = useRouter();
  const { profile, user, profileLoading, loading } = useAuth();

  const email = user?.email ?? undefined;
  const memberSinceIso =
    (profile as { created_at?: string } | null)?.created_at ?? user?.created_at;
  const memberSince =
    memberSinceIso ? new Date(memberSinceIso).toLocaleDateString() : undefined;

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.log('Error signing out:', error);
    // router.replace('/'); // no need if using Link
  };

  const isBusy = loading || profileLoading;

  return (
    <ThemedSafeAreaView>
      <View className="flex-1 px-6 py-10 bg-background-50">
        {/* Profile Header */}
        <View className="items-center mb-10">
          <Image
            source={{
              uri:
                'https://ui-avatars.com/api/?name=' +
                encodeURIComponent(profile?.username || email || 'User'),
            }}
            className="w-24 h-24 rounded-full mb-4"
          />
          <Text className="text-xl font-bodyBold text-primary-600">
            {profile?.username || 'Guest User'}
          </Text>
          <Text className="text-sm text-gray-500">
            {isBusy ? 'Loading...' : email ?? 'No email'}
          </Text>
        </View>

        {/* Info Section */}
        <View className="space-y-4 bg-white rounded-2xl shadow-md p-6 mb-10">
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Member since</Text>
            <Text className="font-bodyBold text-black">
              {isBusy ? 'Loading...' : memberSince ?? 'N/A'}
            </Text>
          </View>
          <View className="h-[1px] bg-gray-200" />
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Status</Text>
            <Text className="font-bodyBold text-green-600">Active</Text>
          </View>
        </View>

        {/* Logout Button using Link */}
        <Link href="/" replace asChild>
          <Pressable
            onPress={handleLogout}
            className="bg-red-500 py-3 rounded-2xl shadow-md"
          >
            <Text className="font-bodyBold text-background-0 text-center">
              Log out
            </Text>
          </Pressable>
        </Link>
      </View>
    </ThemedSafeAreaView>
  );
}
