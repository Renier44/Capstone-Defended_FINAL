import React, { useState, useCallback } from 'react';
import * as SecureStore from 'expo-secure-store';
import { FontAwesome, FontAwesome5, Ionicons, MaterialIcons } from '@expo/vector-icons';
import { useRouter, useFocusEffect } from 'expo-router';
import {
    Image,
    SafeAreaView,
    ScrollView,
    StatusBar,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';

// Local images (These should be dynamic in a real app but are fine for a demo)
// Assuming these are loaded correctly via your asset bundler setup
import clinic1Image from '../assets/images/clinic1.jpg';
import clinic2Image from '../assets/images/clinic2.jpg';
import clinic3Image from '../assets/images/clinic3.jpg';
import clinic4Image from '../assets/images/clinic4.jpg';
import clinic5Image from '../assets/images/clinic5.jpg';
import clinic6Image from '../assets/images/clinic6.jpg';
import dr1Image from '../assets/images/dr1.jpg';
import dr2Image from '../assets/images/dr2.jpg';

export default function DashboardScreen() {
    const router = useRouter();

    const [userData, setUserData] = useState({ name: '', email: '', profile_image: null });

    // Function to load the profile data from SecureStore, wrapped in useCallback
    const loadLocalUserProfile = useCallback(async () => {
        try {
            // Retrieve the complete userProfile data saved by the login or profile screen
            const localProfileStr = await SecureStore.getItemAsync('userProfile');
            if (localProfileStr) {
                const localProfile = JSON.parse(localProfileStr);
                
                // Load the image URI directly from the locally stored profile data
                setUserData({
                    name: localProfile.name || 'Guest',
                    email: localProfile.email || '',
                    // This line trusts the value stored in SecureStore
                    profile_image: localProfile.profile_image || null, 
                });
            } else {
                // If profile data is missing, check if the user is truly logged out
                const token = await SecureStore.getItemAsync('userToken');
                if (!token) {
                    router.replace('/login');
                } else {
                    // Profile data missing but token exists (set to default)
                    setUserData({ name: 'Guest', email: '', profile_image: null });
                }
            }
        } catch (error) {
            console.warn('Failed to load local user profile:', error);
            const token = await SecureStore.getItemAsync('userToken');
            if (!token) {
                router.replace('/login');
            }
        }
    }, [router]);

    // useFocusEffect is CRITICAL: it ensures data is reloaded whenever the screen is visible.
    useFocusEffect(
        useCallback(() => {
            loadLocalUserProfile();
            return () => {}; 
        }, [loadLocalUserProfile])
    );

    const handleBookAppointment = () => router.push('/book-appointment');
    const handleDoctorsPress = () => router.push('/doctors');
    const handleProfilePress = () => router.push('/profile');
    const handleAppointmentsPress = () => router.push('/my-appointments');
    
    // Function to handle navigation to the notifications screen
    const handleNotificationsPress = () => router.push('/notification');
    
    const handleBackPress = () => router.replace('/');

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="dark-content" />

            <ScrollView contentContainerStyle={styles.scrollViewContent}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity style={styles.circleBackButton} onPress={handleBackPress}>
                        <Ionicons name="arrow-back" size={20} color="#000" />
                    </TouchableOpacity>

                    <View style={styles.userInfo}>
                        {/* PROFILE IMAGE DISPLAY - This image source is updated on focus */}
                        <Image
                            source={{
                                uri:
                                    // Use the stored URI if it exists
                                    userData.profile_image
                                        ? userData.profile_image
                                        : 'https://cdn-icons-png.flaticon.com/512/706/706830.png',
                            }}
                            style={styles.profilePic}
                        />
                        <View style={styles.userNameRow}>
                            <View>
                                <Text style={styles.greetingText}>Hi, Welcome Back</Text>
                                <Text style={styles.userName}>{userData.name || 'Guest'}</Text>
                            </View>

                            <View style={styles.headerIcons}>
                                {/* Notifications Icon (In Header) */}
                                <TouchableOpacity style={styles.iconButton} onPress={handleNotificationsPress}>
                                    <MaterialIcons name="notifications" size={24} color="#555" />
                                </TouchableOpacity>
                                <TouchableOpacity style={styles.iconButton}>
                                    <Ionicons name="settings" size={24} color="#555" />
                                </TouchableOpacity>
                            </View>
                        </View>
                    </View>

                    <View style={styles.locationInfo}>
                        <MaterialIcons name="location-on" size={16} color="#888" />
                        <Text style={styles.locationText}>Limketkai, Lapasan</Text>
                    </View>
                </View>

                {/* Services Section */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.sectionTitle}>Services</Text>
                </View>
                <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.serviceList}>
                    {/* Preliminary Eye Screening button with correct onPress function */}
                    <TouchableOpacity style={styles.serviceItem} onPress={() => router.push('/eye-screening')}>
                        <View style={[styles.serviceIconContainer, { backgroundColor: '#FF8888' }]}>
                            <FontAwesome name="eye" size={30} color="#fff" />
                        </View>
                        <Text style={styles.serviceText}>Preliminary Eye Screening</Text>
                    </TouchableOpacity>
                    {/* Book Appointment Only button with correct onPress function */}
                    <TouchableOpacity style={styles.serviceItem} onPress={handleBookAppointment}>
                        <View style={[styles.serviceIconContainer, { backgroundColor: '#77CDE0' }]}>
                            <FontAwesome5 name="calendar-check" size={30} color="#fff" />
                        </View>
                        <Text style={styles.serviceText}>Book Appointment Only</Text>
                    </TouchableOpacity>
                </ScrollView>

                {/* Available Doctors Section */}
                <View style={styles.sectionHeader}>
                    <Text style={styles.sectionTitle}>Available Doctors</Text>
                    <TouchableOpacity onPress={handleDoctorsPress}>
                        <Text style={styles.seeAllText}>See All</Text>
                    </TouchableOpacity>
                </View>
                <View style={styles.doctorList}>
                    <TouchableOpacity style={styles.doctorCard}>
                        <Image source={dr2Image} style={styles.doctorImage} />
                        <View style={styles.doctorInfo}>
                            <Text style={styles.doctorName}>Dr. Mikaela Sherry Lopez</Text>
                            <Text style={styles.doctorAvailability}>Mon-Fri (10AM - 9PM)</Text>
                        </View>
                        <Text style={styles.doctorSpecialty}>Optometrist</Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.doctorCard}>
                        <Image source={dr1Image} style={styles.doctorImage} />
                        <View style={styles.doctorInfo}>
                            <Text style={styles.doctorName}>Dr. Maria Cherry Lopez</Text>
                            <Text style={styles.doctorAvailability}>Mon-Fri (10AM - 9PM)</Text>
                        </View>
                        <Text style={styles.doctorSpecialty}>Optometrist</Text>
                    </TouchableOpacity>
                </View>

                {/* Clinics Overview */}
                <View style={styles.clinicsOverviewContainer}>
                    <View style={styles.sectionHeader}>
                        <Text style={styles.clinicsTitle}>Clinics Overview</Text>
                        <TouchableOpacity onPress={() => router.push('/packages')}>
                            <Text style={styles.seeAllText}>Our Packages</Text>
                        </TouchableOpacity>
                    </View>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.clinicList}>
                        {[clinic1Image, clinic2Image, clinic3Image, clinic4Image, clinic5Image, clinic6Image].map((img, index) => (
                            <TouchableOpacity key={index} style={styles.aestheticClinicCard}>
                                <Image source={img} style={styles.aestheticClinicImage} />
                                <View style={styles.cardContent}>
                                    <Text style={styles.aestheticClinicName}>Clinic {index + 1}</Text>
                                    <Text style={styles.clinicLocation}>Cagayan de Oro City</Text>
                                </View>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                </View>
            </ScrollView>

            {/* Bottom Nav - UPDATED ORDER: Home, Appointments, Notifications, Profile */}
            <View style={styles.bottomNavigation}>
                {/* 1. HOME (Active) */}
                <TouchableOpacity style={styles.navItem}>
                    <MaterialIcons name="home" size={24} color="#65A3D5" />
                    <Text style={[styles.navText, { color: '#65A3D5' }]}>Home</Text>
                </TouchableOpacity>
                
                {/* 2. APPOINTMENTS (Non-active color #888) */}
                <TouchableOpacity style={styles.navItem} onPress={handleAppointmentsPress}>
                    <MaterialIcons name="event-note" size={24} color="#888" />
                    <Text style={styles.navText}>Appointments</Text>
                </TouchableOpacity>

                {/* 3. NOTIFICATIONS (Replaced History, Non-active color #888) */}
                <TouchableOpacity style={styles.navItem} onPress={handleNotificationsPress}>
                    <MaterialIcons name="notifications" size={24} color="#888" />
                    <Text style={styles.navText}>Notifications</Text>
                </TouchableOpacity>
                
                {/* 4. PROFILE (Non-active color #888) */}
                <TouchableOpacity style={styles.navItem} onPress={handleProfilePress}>
                    <MaterialIcons name="person" size={24} color="#888" />
                    <Text style={styles.navText}>Profile</Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#C8EAF7' },
    scrollViewContent: { padding: 20, paddingBottom: 100 },

    circleBackButton: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#eee',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 10,
    },

    header: { marginBottom: 20 },
    userInfo: { flexDirection: 'row', alignItems: 'center', marginVertical: 15 },
    profilePic: {
        width: 80,
        height: 80,
        borderRadius: 40,
        marginRight: 12,
        borderWidth: 3,
        borderColor: '#65A3D5',
    },

    greetingText: { fontSize: 14, color: '#555' },
    userName: { fontSize: 20, fontWeight: 'bold', color: '#222' },

    userNameRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        flex: 1
    },
    headerIcons: { flexDirection: 'row', alignItems: 'center' },
    iconButton: { marginLeft: 15 },

    locationInfo: { flexDirection: 'row', alignItems: 'center' },
    locationText: { fontSize: 14, color: '#555', marginLeft: 5 },

    sectionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 15,
    },
    sectionTitle: { fontSize: 20, fontWeight: 'bold', color: '#333' },
    seeAllText: { fontSize: 14, color: '#65A3D5' },

    serviceList: { marginBottom: 20 },
    serviceItem: {
        alignItems: 'center',
        width: 120,
        marginRight: 15,
    },
    serviceIconContainer: {
        width: 60,
        height: 60,
        borderRadius: 15,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 8,
    },
    serviceText: {
        fontSize: 12,
        fontWeight: '500',
        color: '#555',
        textAlign: 'center',
    },

    doctorList: {
        marginBottom: 20,
    },
    doctorCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#77CDE0',
        borderRadius: 15,
        padding: 15,
        marginBottom: 10,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
    },
    doctorImage: {
        width: 60,
        height: 60,
        borderRadius: 30,
        marginRight: 15,
        borderWidth: 2,
        borderColor: '#fff',
    },
    doctorInfo: {
        flex: 1,
    },
    doctorName: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#fff',
    },
    doctorAvailability: {
        fontSize: 12,
        color: '#fff',
    },
    doctorSpecialty: {
        fontSize: 14,
        color: '#fff',
    },

    // Aesthetic Clinics Overview Styles
    clinicsOverviewContainer: {
        marginBottom: 20,
    },
    clinicsTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#333',
    },
    clinicList: {
        marginTop: 10,
    },
    aestheticClinicCard: {
        width: 250,
        borderRadius: 20,
        overflow: 'hidden',
        marginRight: 15,
        backgroundColor: '#FFD54F',
        elevation: 5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
    },
    aestheticClinicImage: {
        width: '100%',
        height: 150,
        resizeMode: 'cover',
    },
    cardContent: {
        padding: 15,
    },
    aestheticClinicName: {
        fontSize: 16,
        fontWeight: 'bold',
        color: '#333',
    },
    clinicLocation: {
        fontSize: 12,
        color: '#888',
        marginTop: 5,
    },

    bottomNavigation: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        backgroundColor: '#77CDE0', // Reverted to original light blue
        borderTopWidth: 1,
        borderTopColor: '#77CDE0',
        paddingVertical: 10,
        paddingHorizontal: 5,
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        elevation: 8,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -2 },
        shadowOpacity: 0.1,
        shadowRadius: 4
    },
    navItem: { alignItems: 'center', padding: 5 },
    navText: { fontSize: 10, color: '#fff', marginTop: 3 }, // Original default text color
});
