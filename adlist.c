// A generic doubly linked list implementation
#include <stdlib.h>

typedef struct listNode {
    struct listNode *prev;  // 结构体的自引用只能是指针类型
    struct listNode *next;
    void *value;            // void代表可指向任何类型数据
} listNode;

typedef struct list {
    listNode *head;
    listNode *tail;
	void (*free)(void *ptr);  // 函数指针,必须加()
    unsigned long len;
} list;

#define listLength(l) ((l)->len)
#define listNodeValue(n) ((n)->value)

/*
Create a new list. The created list can be freed with AlFreeList(),
but private value of every node need to be freed by the user before to call AlFreeList().
*/
list *listCreate()
{
    list *list = malloc(sizeof(list));
    if (list == NULL) return NULL;

    list->head = list->tail = NULL;
    list->len = 0;
    list->dup = NULL;
    list->free = NULL;
    list->match = NULL;
    return list;
}

void listEmpty(list *list)  // Remove all the elements from the list without destroying the list itself.
{
    unsigned long len = list->len;;
    listNode *current = list->head, *next;
    while(len--) {
        next = current->next;
        if (list->free) list->free(current->value);
        free(current);
        current = next;
    }
    list->head = list->tail = NULL;
    list->len = 0;
}

void listRelease(list *list)  // Free the whole list.This function can't fail.
{
    listEmpty(list);
    free(list);
}

list *listAddNodeHead(list *list, void *value)  // Add a new node to the list, to head, containing the specified 'value' pointer as value.
{
    listNode *node;

    if ((node = malloc(sizeof(*node))) == NULL)
        return NULL;
    node->value = value;
    if (list->len == 0) {
        list->head = list->tail = node;
        node->prev = node->next = NULL;
    }
    else {
        node->prev = NULL;
        node->next = list->head;
        list->head->prev = node;
        list->head = node;
    }
    list->len++;
    return list;
}

list *listAddNodeTail(list *list, void *value)
{
    listNode *node;

    if ((node = malloc(sizeof(*node))) == NULL)
        return NULL;
    node->value = value;
    if (list->len == 0) {
        list->head = list->tail = node;
        node->prev = node->next = NULL;
    }
    else {
        node->prev = list->tail;
        node->next = NULL;
        list->tail->next = node;
        list->tail = node;
    }
    list->len++;
    return list;
}

list *listInsertNode(list *list, listNode *old_node, void *value, int after) {
    listNode *node;

    if ((node = malloc(sizeof(*node))) == NULL)
        return NULL;
    node->value = value;
    if (after) {
        node->prev = old_node;
        node->next = old_node->next;
        if (list->tail == old_node) {
            list->tail = node;
        }
    }
    else {
        node->next = old_node;
        node->prev = old_node->prev;
        if (list->head == old_node) {
            list->head = node;
        }
    }
    if (node->prev != NULL) {  // 使用node判断可统一操作
        node->prev->next = node;
    }
    if (node->next != NULL) {
        node->next->prev = node;
    }
    list->len++;
    return list;
}

void listDelNode(list *list, listNode *node)
{
    if (node->prev)
        node->prev->next = node->next;
    else
        list->head = node->next;
    if (node->next)
        node->next->prev = node->prev;
    else
        list->tail = node->prev;
    if (list->free) list->free(node->value);
    free(node);
    list->len--;
}

void listRotate(list *list) {  /* Rotate the list removing the tail node and inserting it to the head. */
    if (listLength(list) > 1){
        listNode *tail = list->tail;
        /* Detach current tail */
        list->tail = tail->prev;
        list->tail->next = NULL;
        /* Move it as head */
        list->head->prev = tail;
        tail->prev = NULL;
        tail->next = list->head;
        list->head = tail;
    }
}

int main(){
    /*
    如果在程序中定义了一个函数，那么在编译时系统就会为这个函数代码分配一段存储空间，这段存储空间的首地址称为这个函数的地址。而且函数名表示的就是这个地址。
    指向函数的指针变量没有 ++ 和 -- 运算
    */
    printf("%p\n",listEmpty);
}
