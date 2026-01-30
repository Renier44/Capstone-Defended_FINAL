import React, { useState, useEffect, useCallback } from 'react';
import * as SecureStore from 'expo-secure-store';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Image,
  ActivityIndicator,
  RefreshControl,
  Alert,
  Modal,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';
import { Picker } from '@react-native-picker/picker';

// Import your doctor images
import dr1 from '../assets/images/dr1.jpg';
import dr2 from '../assets/images/dr2.jpg';

// Map doctor names to images
const DOCTOR_IMAGE_MAP = {
  'Dr. Mikaela Cherry Lopez': dr2,
  'Dr. Maria Sherry Lopez': dr1,
};

export default function MyAppointmentsScreen() {
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('pending');
  const [cancelModalVisible, setCancelModalVisible] = useState(false);
  const [currentCancelId, setCurrentCancelId] = useState(null);
  const [cancelReason, setCancelReason] = useState('');

  const cancelOptions = [
    'Not available on the date',
    'Not Feeling better',
    'Scheduling conflict',
    'Other',
  ];

  useEffect(() => {
    fetchAppointments();
  }, []);

  const getAuthToken = async () => await SecureStore.getItemAsync('userToken');

  const fetchAppointments = async () => {
    setLoading(true);
    try {
      const token = await getAuthToken();
      if (!token) {
        Alert.alert('Error', 'User not logged in');
        setLoading(false);
        return;
      }

      const API_URL = 'https://capstone-defended-final.onrender.com/api/my-appointments/';
      const response = await fetch(API_URL, {
        headers: { Authorization: `Token ${token}` },
      });

      const text = await response.text();
      try {
        const data = JSON.parse(text);
        setAppointments(data);
      } catch {
        console.error('Failed to parse JSON:', text);
        Alert.alert('Error', 'Failed to fetch appointments. Check your server.');
        setAppointments([]);
      }
    } catch (error) {
      console.error('Error fetching appointments:', error);
      setAppointments([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchAppointments();
  }, []);

  const openCancelModal = (id) => {
    setCurrentCancelId(id);
    setCancelReason(cancelOptions[0]);
    setCancelModalVisible(true);
  };

  const submitCancel = async () => {
    if (!cancelReason.trim()) {
      Alert.alert('Error', 'Reason is required');
      return;
    }
    try {
      const token = await getAuthToken();
      const API_URL = `https://capstone-defended-final.onrender.com/api/cancel-appointment/${currentCancelId}/`;
      const res = await fetch(API_URL, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Token ${token}`,
        },
        body: JSON.stringify({ status: 'Cancelled', cancel_reason: cancelReason }),
      });

      if (res.ok) {
        Alert.alert('Success', 'Appointment cancelled successfully.');
        fetchAppointments();
      } else {
        Alert.alert('Error', 'Failed to cancel appointment.');
      }
    } catch (err) {
      console.error(err);
      Alert.alert('Error', 'Something went wrong.');
    } finally {
      setCancelModalVisible(false);
      setCancelReason('');
      setCurrentCancelId(null);
    }
  };

  const deleteAppointment = async (id) => {
    Alert.alert('Confirm', 'Delete this appointment permanently?', [
      { text: 'No' },
      {
        text: 'Yes',
        onPress: async () => {
          try {
            const token = await getAuthToken();
            const API_URL = `https://capstone-defended-final.onrender.com/api/delete-appointment/${id}/`;
            const res = await fetch(API_URL, {
              method: 'DELETE',
              headers: { Authorization: `Token ${token}` },
            });

            if (res.ok) {
              Alert.alert('Deleted', 'Appointment removed successfully.');
              fetchAppointments();
            } else {
              Alert.alert('Error', 'Failed to delete appointment.');
            }
          } catch (err) {
            console.error(err);
            Alert.alert('Error', 'Something went wrong.');
          }
        },
      },
    ]);
  };

  const filteredAppointments = appointments.filter((item) => {
    const status = item.status?.toLowerCase() || '';
    if (filter === 'pending') return status === 'pending';
    if (filter === 'upcoming') return ['scheduled', 'confirmed'].includes(status);
    if (filter === 'cancelled') return status === 'cancelled';
    if (filter === 'past') return status === 'completed';
    return false;
  });

  const renderAppointmentCard = (item) => {
    const doctorImage = DOCTOR_IMAGE_MAP[item.doctor_name] || {
      uri: `https://ui-avatars.com/api/?name=${encodeURIComponent(item.doctor_name || 'Doctor')}&background=2260FF&color=fff`,
    };

    const status = item.status?.toLowerCase() || '';
    const isPending = ['pending', 'scheduled', 'confirmed'].includes(status);
    const isCancelled = status === 'cancelled';
    const isCompleted = status === 'completed';

    return (
      <View style={styles.card}>
        {/* Doctor Image + Name */}
        <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10 }}>
          <Image source={doctorImage} style={styles.avatar} />
          <View style={{ marginLeft: 12 }}>
            <Text style={styles.name}>{item.doctor_name}</Text>
            <Text style={styles.specialty}>Optometrist</Text>
          </View>
        </View>

        {/* Patient Info */}
        <Text style={styles.detailText}>👤 {item.firstName} {item.lastName}</Text>
        <Text style={styles.detailText}>🎂 {item.age} yrs | {item.gender}</Text>
        <Text style={styles.detailText}>📌 Booking For: {item.bookingFor}</Text>

        {item.preliminary_result ? (
          <Text style={styles.detailText}>🧪 Preliminary Result: {item.preliminary_result}</Text>
        ) : (
          <Text style={styles.detailText}>📝 Reason: {item.reason}</Text>
        )}

        <View style={styles.row}>
          <Ionicons name="calendar-outline" size={16} color="#2260FF" />
          <Text style={styles.detailText}> {item.date}</Text>
        </View>
        <View style={styles.row}>
          <Ionicons name="time-outline" size={16} color="#2260FF" />
          <Text style={styles.detailText}> {item.time}</Text>
        </View>

        {isCancelled && item.cancel_reason && (
          <Text style={[styles.detailText, { fontStyle: 'italic', color: '#E53935' }]}>
            ❌ Cancel Reason: {item.cancel_reason}
          </Text>
        )}

        {/* Action Buttons */}
        <View style={styles.actions}>
          {isPending && (
            <TouchableOpacity style={styles.btnCancel} onPress={() => openCancelModal(item.id)}>
              <Text style={styles.btnText}>Cancel</Text>
            </TouchableOpacity>
          )}

          {(isCancelled || isCompleted) && (
            <TouchableOpacity style={styles.btnDelete} onPress={() => deleteAppointment(item.id)}>
              <Text style={styles.btnText}>Delete</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Pending Badge */}
        {status === 'pending' && (
          <View style={styles.pendingBadge}>
            <Text style={styles.pendingText}>Pending</Text>
          </View>
        )}
      </View>
    );
  };

  if (loading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color="#2260FF" />
        <Text style={{ color: '#2260FF', marginTop: 10 }}>Loading appointments...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={26} color="#2260FF" />
        </TouchableOpacity>
        <Text style={styles.headerText}>My Appointments</Text>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {['pending','upcoming','cancelled','past'].map((t) => (
          <TouchableOpacity key={t} style={[styles.tab, filter === t && styles.activeTab]} onPress={() => setFilter(t)}>
            <Text style={[styles.tabText, filter === t && styles.activeTabText]}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Appointment List */}
      {filteredAppointments.length === 0 ? (
        <Text style={styles.emptyText}>No {filter} appointments found.</Text>
      ) : (
        <FlatList
          data={filteredAppointments}
          keyExtractor={(item) => item.id.toString()}
          renderItem={({ item }) => renderAppointmentCard(item)}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        />
      )}

      {/* Cancel Modal */}
      <Modal visible={cancelModalVisible} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalBox}>
            <Text style={styles.modalTitle}>Cancel Appointment</Text>
            <Text style={styles.modalText}>Select a reason for cancelling:</Text>

            <View style={{ borderWidth: 1, borderColor: '#ccc', borderRadius: 10, height: 50, justifyContent: 'center', marginBottom: 15 }}>
              <Picker
                selectedValue={cancelReason}
                onValueChange={(value) => setCancelReason(value)}
                mode="dropdown"
              >
                {cancelOptions.map((option) => (
                  <Picker.Item key={option} label={option} value={option} />
                ))}
              </Picker>
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity style={[styles.modalBtn, { backgroundColor: '#2260FF' }]} onPress={submitCancel}>
                <Text style={styles.modalBtnText}>Submit</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.modalBtn, { backgroundColor: '#757575' }]} onPress={() => setCancelModalVisible(false)}>
                <Text style={styles.modalBtnText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#D9ECFF' },
  header: { flexDirection: 'row', alignItems: 'center', padding: 20, marginTop: 40 },
  headerText: { flex: 1, textAlign: 'center', fontSize: 20, fontWeight: '700', color: '#2260FF' },
  loader: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  tabs: { flexDirection: 'row', justifyContent: 'space-around', marginHorizontal: 16, marginBottom: 12 },
  tab: { paddingVertical: 8, paddingHorizontal: 18, borderRadius: 20, backgroundColor: '#E8EAF6' },
  activeTab: { backgroundColor: '#77CDE0' },
  tabText: { fontSize: 14, color: '#555', fontWeight: '500' },
  activeTabText: { color: '#fff', fontWeight: '700' },
  card: { 
    backgroundColor: '#fff', 
    borderRadius: 16, 
    marginHorizontal: 16, 
    marginVertical: 10, 
    padding: 16, 
    flexDirection: 'column', 
    alignItems: 'flex-start', 
    shadowColor: '#000', 
    shadowOpacity: 0.1, 
    shadowOffset: { width: 0, height: 2 }, 
    shadowRadius: 6, 
    elevation: 2,
    borderLeftWidth: 6, 
    borderLeftColor: '#77CDE0',
    paddingLeft: 10,
  },
  avatar: { width: 55, height: 55, borderRadius: 30 },
  name: { fontSize: 16, fontWeight: '700', color: '#2260FF' },
  specialty: { fontSize: 13, color: '#333' },
  detailText: { fontSize: 13, color: '#000', marginVertical: 2 },
  row: { flexDirection: 'row', alignItems: 'center', marginTop: 3 },
  actions: { flexDirection: 'row', marginTop: 10, gap: 8 },
  btnCancel: { backgroundColor: '#E53935', paddingHorizontal: 16, paddingVertical: 6, borderRadius: 12 },
  btnDelete: { backgroundColor: '#757575', paddingHorizontal: 16, paddingVertical: 6, borderRadius: 12 },
  btnText: { color: '#fff', fontWeight: '600', fontSize: 13 },
  pendingBadge: { marginTop: 10, alignSelf: 'flex-start', backgroundColor: '#FFB300', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 10 },
  pendingText: { color: '#fff', fontWeight: '700', fontSize: 12 },
  emptyText: { textAlign: 'center', marginTop: 40, color: '#2260FF', fontSize: 16, fontWeight: '600' },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', alignItems: 'center' },
  modalBox: { width: '85%', backgroundColor: '#fff', borderRadius: 16, padding: 20 },
  modalTitle: { fontSize: 18, fontWeight: '700', marginBottom: 12, color: '#2260FF' },
  modalText: { fontSize: 14, marginBottom: 10 },
  modalButtons: { flexDirection: 'row', justifyContent: 'space-between' },
  modalBtn: { flex: 1, marginHorizontal: 5, paddingVertical: 10, borderRadius: 10, alignItems: 'center' },
  modalBtnText: { color: '#fff', fontWeight: '600' },
});
